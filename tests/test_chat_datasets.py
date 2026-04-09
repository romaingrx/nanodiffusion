from pathlib import Path

import pytest
from datasets import Dataset

from nanodiffusion.data.chat_datasets import (
    CHAT_DATASETS,
    ChatDatasetFactory,
    HuggingFaceChatSource,
    JsonlChatSource,
    get_chat_dataset,
    normalize_message,
    register_chat,
)
from nanodiffusion.data.chat_datasets.gsm8k import (
    row_to_conversation as gsm8k_row_to_conversation,
)
from nanodiffusion.data.chat_datasets.gsm8k import strip_tool_calls
from nanodiffusion.data.chat_datasets.smoltalk import (
    row_to_conversation as smoltalk_row_to_conversation,
)
from nanodiffusion.data.chat_source import ChatSource
from nanodiffusion.data.datasets import DownloadOptions


def _smoltalk_dataset_fixture() -> Dataset:
    """Tiny in-memory Dataset matching SmolTalk's on-disk schema.

    ``Dataset.from_dict`` lets us build a fake HuggingFace dataset
    without touching the network — the same surface ``load_dataset``
    would return so ``HuggingFaceChatSource`` can be exercised directly.
    """
    return Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "system", "content": "be helpful"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                ],
            ],
        }
    )


def test_huggingface_chat_source_len_and_getitem() -> None:
    src = HuggingFaceChatSource(
        _smoltalk_dataset_fixture(), smoltalk_row_to_conversation
    )
    assert len(src) == 2
    first = src[0]
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][-1]["role"] == "assistant"


def test_huggingface_chat_source_rejects_empty_dataset() -> None:
    empty = Dataset.from_dict({"messages": []})
    with pytest.raises(ValueError, match="empty dataset"):
        HuggingFaceChatSource(empty, smoltalk_row_to_conversation)


def test_smoltalk_row_preserves_system_message_for_later_merge() -> None:
    """render_conversation handles the system merge; the source must
    not drop it on the way in or that training signal is lost silently.
    """
    row = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ],
    }
    conv = smoltalk_row_to_conversation(row)
    roles = [m["role"] for m in conv["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_smoltalk_row_raises_on_short_conversation() -> None:
    row = {"messages": [{"role": "user", "content": "solo"}]}
    with pytest.raises(ValueError, match="fewer than 2"):
        smoltalk_row_to_conversation(row)


def testnormalize_message_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="unsupported role"):
        normalize_message({"role": "tool", "content": "x"})


def testnormalize_message_rejects_non_string_content() -> None:
    with pytest.raises(TypeError, match="bad content type"):
        normalize_message({"role": "user", "content": 42})


def test_gsm8k_row_strips_tool_markers() -> None:
    row = {
        "question": "Weng earns $12/hour. For 50 minutes, how much?",
        "answer": (
            "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\n"
            "Working 50 minutes, she earned 0.2 * 50 = $<<0.2*50=10>>10.\n"
            "#### 10"
        ),
    }
    conv = gsm8k_row_to_conversation(row)
    assistant = conv["messages"][-1]
    assert "<<" not in assistant["content"]
    assert ">>" not in assistant["content"]
    # The surrounding prose plus the final marker survive stripping, so
    # any downstream grader that keys off ``#### <number>`` still works.
    assert "#### 10" in assistant["content"]


def test_gsm8k_flattening_does_not_eat_bare_delimiters() -> None:
    """Anchoring on ``=`` inside the delimiters prevents literal ``<<`` /
    ``>>`` in natural text from being silently consumed.
    """
    answer = "The quoted snippet was '<<not a tool call>> says the book'. #### 0"
    stripped = strip_tool_calls(answer)
    assert stripped == answer


def test_gsm8k_flattening_handles_empty_expression() -> None:
    """A truly broken marker should still strip cleanly, not raise."""
    answer = "Some weird <<=>> marker. #### 1"
    stripped = strip_tool_calls(answer)
    assert "<<" not in stripped
    assert "#### 1" in stripped


def test_jsonl_chat_source_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "identity.jsonl"
    content = (
        '[{"role": "user", "content": "who are you?"},'
        ' {"role": "assistant", "content": "I am nanodiffusion."}]\n'
        '[{"role": "user", "content": "what day is it?"},'
        ' {"role": "assistant", "content": "I do not know."}]\n'
    )
    path.write_text(content)
    src = JsonlChatSource(path)
    assert len(src) == 2
    assert src[0]["messages"][1]["content"] == "I am nanodiffusion."


def test_jsonl_chat_source_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "identity.jsonl"
    path.write_text(
        '\n\n[{"role":"user","content":"hi"},{"role":"assistant","content":"hey"}]\n\n'
    )
    src = JsonlChatSource(path)
    assert len(src) == 1


def test_jsonl_chat_source_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        JsonlChatSource(tmp_path / "nope.jsonl")


def test_jsonl_chat_source_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")
    with pytest.raises(ValueError, match="empty"):
        JsonlChatSource(path)


def test_jsonl_chat_source_rejects_non_list_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"role":"user","content":"hi"}\n')
    with pytest.raises(TypeError, match="list"):
        JsonlChatSource(path)


def test_chat_dataset_registry_has_expected_entries() -> None:
    assert set(CHAT_DATASETS) >= {"smoltalk", "gsm8k", "identity"}


def test_get_chat_dataset_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown chat dataset"):
        get_chat_dataset("nope")


def _dummy_factory(
    data_dir: Path,
    *,
    download: bool = True,
    download_options: DownloadOptions | None = None,
) -> ChatSource:  # pragma: no cover - registration-only stub
    raise NotImplementedError(data_dir, download, download_options)


def test_register_chat_rejects_duplicates() -> None:
    _: ChatDatasetFactory = _dummy_factory  # static assurance of the protocol fit
    with pytest.raises(ValueError, match="already registered"):
        register_chat("smoltalk")(_dummy_factory)


def test_identity_factory_uses_existing_file(tmp_path: Path) -> None:
    """Pre-create the JSONL so the factory skips the network fetch.

    Directly validates the offline path of :func:`_identity_factory`
    without touching S3.
    """
    (tmp_path / "identity_conversations.jsonl").write_text(
        '[{"role":"user","content":"hi"},{"role":"assistant","content":"hey"}]\n'
    )
    factory = get_chat_dataset("identity")
    src = factory(tmp_path, download=False)
    assert len(src) == 1
