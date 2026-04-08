"""Named chat datasets for SFT.

Mirrors :mod:`nanodiffusion.data.datasets` but for indexable chat
conversation sources consumed by the SFT loader. HuggingFace-hosted
datasets are loaded via the :mod:`datasets` library so caching, splits,
and schema decoding stay shared with the broader ML ecosystem; the
identity conversations bundle is a one-off JSONL on Karpathy's S3 bucket
and keeps a direct HTTP fetch.

Three datasets are registered up front (ROM-17):

* ``smoltalk`` — ``HuggingFaceTB/smol-smoltalk`` ``train`` split, general
  multi-turn conversations.
* ``gsm8k`` — ``openai/gsm8k`` ``main`` / ``train``. Calculator tool-call
  markers ``<<expr=result>>`` in the gold answer are stripped to plain
  text (anchored on ``=`` inside the delimiters) so the existing
  string-content ``Message`` schema handles them. Full multi-part
  tool-call support is deferred to a follow-up issue.
* ``identity`` — small JSONL of identity conversations hosted on
  Karpathy's public S3 bucket, same as nanochat.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

from nanodiffusion.chat import Conversation, Message
from nanodiffusion.data.chat_source import ChatSource
from nanodiffusion.data.datasets import DownloadOptions, download_with_backoff

if TYPE_CHECKING:
    from datasets import Dataset

logger = structlog.get_logger(__name__)


@runtime_checkable
class ChatDatasetFactory(Protocol):
    """Builds a :class:`ChatSource` from a local cache directory.

    Mirrors :class:`nanodiffusion.data.datasets.DatasetFactory` — same
    ``(data_dir, download, download_options)`` triplet — so the same
    CLI shape can drive both pretrain and SFT downloads.
    """

    def __call__(
        self,
        data_dir: Path,
        *,
        download: bool = True,
        download_options: DownloadOptions | None = None,
    ) -> ChatSource: ...


CHAT_DATASETS: dict[str, ChatDatasetFactory] = {}


def register_chat(name: str) -> Callable[[ChatDatasetFactory], ChatDatasetFactory]:
    def decorator(fn: ChatDatasetFactory) -> ChatDatasetFactory:
        if name in CHAT_DATASETS:
            msg = f"Chat dataset {name!r} already registered"
            raise ValueError(msg)
        CHAT_DATASETS[name] = fn
        return fn

    return decorator


def get_chat_dataset(name: str) -> ChatDatasetFactory:
    if name not in CHAT_DATASETS:
        available = ", ".join(sorted(CHAT_DATASETS)) or "(none)"
        msg = f"Unknown chat dataset {name!r}. Available: {available}"
        raise KeyError(msg)
    return CHAT_DATASETS[name]


type RowToConversation = Callable[[dict[str, object]], Conversation]


class HuggingFaceChatSource:
    """Wraps a :class:`datasets.Dataset` as a :class:`ChatSource`.

    Schema-specific decoding — e.g. GSM8K's tool-call flattening — lives
    in a caller-supplied ``row_to_conversation`` callable so this class
    stays dataset-agnostic. The wrapped ``Dataset`` handles caching,
    random access, and the underlying parquet read so neither this class
    nor the chat-loader has to reimplement that machinery.

    ``row_to_conversation`` must not return ``None``: rows that cannot be
    decoded should raise so misconfigured datasets fail loudly at load
    time rather than silently dropping training signal.
    """

    def __init__(
        self,
        dataset: "Dataset",
        row_to_conversation: RowToConversation,
    ) -> None:
        if len(dataset) == 0:
            err = "HuggingFaceChatSource wraps an empty dataset"
            raise ValueError(err)
        self._dataset = dataset
        self._row_to_conversation = row_to_conversation

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Conversation:
        return self._row_to_conversation(self._dataset[index])


class JsonlChatSource:
    """Reads one conversation per line from a JSONL file.

    Each line must be a JSON array of ``{role, content}`` dicts matching
    :class:`Message`. Used for the small identity conversations bundle,
    which isn't on the HuggingFace hub.
    """

    def __init__(self, path: Path) -> None:
        if not path.exists():
            err = f"JsonlChatSource file not found: {path}"
            raise FileNotFoundError(err)
        conversations: list[Conversation] = []
        with path.open(encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                messages = json.loads(line)
                if not isinstance(messages, list):
                    err = f"JsonlChatSource expected list, got {type(messages)}"
                    raise TypeError(err)
                conversations.append({"messages": messages})
        if not conversations:
            err = f"JsonlChatSource file is empty: {path}"
            raise ValueError(err)
        self._conversations = conversations

    def __len__(self) -> int:
        return len(self._conversations)

    def __getitem__(self, index: int) -> Conversation:
        return self._conversations[index]


def _normalize_message(raw: object) -> Message:
    """Coerce a dataset row entry into a :class:`Message`.

    Raises on missing fields or unsupported roles rather than returning
    ``None`` — a single bad row should abort the load so the caller
    notices and fixes the schema drift instead of training on a silently
    truncated dataset.
    """
    if not isinstance(raw, dict):
        err = f"Expected dict message, got {type(raw).__name__}"
        raise TypeError(err)
    role = raw.get("role")
    content = raw.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        err = f"Message has bad role/content types: {raw!r}"
        raise TypeError(err)
    if role not in {"user", "assistant", "system"}:
        err = f"Message has unsupported role {role!r}"
        raise ValueError(err)
    return {"role": role, "content": content}  # pyright: ignore[reportReturnType]


def _smoltalk_row_to_conversation(row: dict[str, object]) -> Conversation:
    """Decode a SmolTalk row into a :class:`Conversation`.

    SmolTalk rows use a ``messages`` column holding a list of
    ``{role, content}`` dicts. An optional leading ``system`` message is
    left in place here; :func:`nanodiffusion.chat.render_conversation`
    merges it into the first user message at render time.
    """
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        err = f"SmolTalk row missing 'messages' list: {row!r}"
        raise TypeError(err)
    messages = [_normalize_message(m) for m in raw_messages]
    if len(messages) < 2:  # noqa: PLR2004  # need at least one turn
        err = f"SmolTalk row has fewer than 2 messages: {messages!r}"
        raise ValueError(err)
    return {"messages": messages}


# Calculator tool-call marker used by GSM8K. Must require ``=`` inside
# the delimiters so bare ``<<`` / ``>>`` in natural text is not silently
# consumed by the flattening regex (see tests/test_chat_datasets.py for
# the negative case).
_GSM8K_TOOL_RE = re.compile(r"<<[^<>]*=[^<>]*>>")


def _strip_gsm8k_tool_calls(answer: str) -> str:
    """Drop ``<<expr=result>>`` markers, keep the surrounding prose.

    GSM8K's gold answers interleave a human-readable solution with
    calculator tool calls. The surrounding text already contains both
    the expression and the result (e.g. "12/60 = $0.2 per minute"), so
    stripping the markers preserves answer correctness while avoiding
    a tool-call vocabulary change.
    """
    return _GSM8K_TOOL_RE.sub("", answer)


def _gsm8k_row_to_conversation(row: dict[str, object]) -> Conversation:
    """Decode a GSM8K row into a simple two-turn conversation."""
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        err = f"GSM8K row has bad question/answer types: {row!r}"
        raise TypeError(err)
    cleaned = _strip_gsm8k_tool_calls(answer)
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": cleaned},
        ],
    }


@dataclass(frozen=True, slots=True)
class _HFChatDataset:
    """Static description of an HF-hosted chat dataset.

    Factories close over these to produce the per-call
    :class:`ChatSource`; the registry stores the factories, not the
    descriptions.
    """

    repo_id: str
    subset: str | None
    split: str
    row_to_conversation: RowToConversation
    doc: str


_SMOLTALK = _HFChatDataset(
    repo_id="HuggingFaceTB/smol-smoltalk",
    subset=None,
    split="train",
    row_to_conversation=_smoltalk_row_to_conversation,
    doc="smol-smoltalk (HuggingFaceTB). ~460K general multi-turn chats.",
)

_GSM8K = _HFChatDataset(
    repo_id="openai/gsm8k",
    subset="main",
    split="train",
    row_to_conversation=_gsm8k_row_to_conversation,
    doc="GSM8K (openai/gsm8k main). Tool-call markers flattened to plain text.",
)


def _make_hf_chat_factory(spec: _HFChatDataset) -> ChatDatasetFactory:
    def factory(
        data_dir: Path,
        *,
        download: bool = True,  # noqa: ARG001 -- datasets lib caches automatically
        download_options: DownloadOptions | None = None,  # noqa: ARG001
    ) -> ChatSource:
        # Deferred import: the datasets library is heavy and we don't
        # want it pulled into process memory when the user only touches
        # the registry (e.g. ``nanodiffusion data list``).
        from datasets import load_dataset  # noqa: PLC0415

        data_dir.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(
            spec.repo_id,
            spec.subset,
            split=spec.split,
            cache_dir=str(data_dir),
        )
        return HuggingFaceChatSource(dataset, spec.row_to_conversation)  # pyright: ignore[reportArgumentType]

    factory.__doc__ = spec.doc
    return factory


_IDENTITY_URL = (
    "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
)
_IDENTITY_FILENAME = "identity_conversations.jsonl"


def _identity_factory(
    data_dir: Path,
    *,
    download: bool = True,
    download_options: DownloadOptions | None = None,
) -> ChatSource:
    """Identity conversations JSONL, same file nanochat uses.

    Not on HuggingFace, so this one still goes through the custom
    retry/backoff downloader in :mod:`nanodiffusion.data.datasets`.
    """
    target = data_dir / _IDENTITY_FILENAME
    if download and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        download_with_backoff(
            _IDENTITY_URL,
            target,
            options=download_options or DownloadOptions(),
        )
    return JsonlChatSource(target)


_identity_factory.__doc__ = (
    "Identity conversations bundle from karpathy-public S3. ~1K rows, "
    "plain JSONL of message lists."
)


for _name, _spec in [("smoltalk", _SMOLTALK), ("gsm8k", _GSM8K)]:
    register_chat(_name)(_make_hf_chat_factory(_spec))

register_chat("identity")(_identity_factory)
