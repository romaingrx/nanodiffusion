import pytest

from nanodiffusion.chat import (
    Conversation,
    SequenceBuilder,
    render_conversation,
    render_for_completion,
)
from nanodiffusion.tokenizer import SpecialToken, Tokenizer


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer()


def test_builder_special_and_text(tok: Tokenizer) -> None:
    b = SequenceBuilder(tok)
    b.special(SpecialToken.BOS)
    b.text("hi")
    b.special(SpecialToken.EOS)

    assert b.ids[0] == tok.bos_token_id
    assert b.ids[-1] == tok.eos_token_id
    assert b.mask == [0] * len(b.ids)


def test_builder_user_message_not_supervised(tok: Tokenizer) -> None:
    b = SequenceBuilder(tok)
    b.message({"role": "user", "content": "hello"})

    assert b.mask == [0] * len(b.ids)


def test_builder_assistant_message_supervised(tok: Tokenizer) -> None:
    b = SequenceBuilder(tok)
    b.message({"role": "assistant", "content": "hi"})

    assert b.mask[0] == 0
    assert all(m == 1 for m in b.mask[1:])


def test_render_conversation_masks(tok: Tokenizer) -> None:
    conv: Conversation = {
        "messages": [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ],
    }
    ids, mask = render_conversation(tok, conv)

    assert len(ids) == len(mask)
    assert mask[0] == 0
    assert mask[-1] == 0
    assert 1 in mask


def test_render_conversation_roundtrip_text(tok: Tokenizer) -> None:
    conv: Conversation = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    ids, _ = render_conversation(tok, conv)
    decoded = tok.decode(ids)
    assert "hi" in decoded
    assert "hello" in decoded


def test_render_conversation_unknown_role_raises(tok: Tokenizer) -> None:
    conv = {"messages": [{"role": "tool", "content": "nope"}]}  # type: ignore[typeddict-item]
    with pytest.raises(ValueError, match="must alternate"):
        render_conversation(tok, conv)


def test_system_message_is_merged_into_first_user(tok: Tokenizer) -> None:
    """A leading system message is prepended to the first user content."""
    conv: Conversation = {
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    ids, _mask = render_conversation(tok, conv)
    decoded = tok.decode(ids)
    # Both pieces survive, joined by the \n\n separator nanochat uses.
    assert "be helpful" in decoded
    assert "hi" in decoded
    assert "be helpful\n\nhi" in decoded


def test_system_message_merge_does_not_mutate_input(tok: Tokenizer) -> None:
    conv: Conversation = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    render_conversation(tok, conv)
    # The original list must be untouched so callers can reuse conversations.
    assert conv["messages"][0]["role"] == "system"
    assert conv["messages"][1]["content"] == "hi"


def test_system_message_without_following_user_raises(tok: Tokenizer) -> None:
    conv = {"messages": [{"role": "system", "content": "be helpful"}]}  # type: ignore[typeddict-item]
    with pytest.raises(ValueError, match="System message must be followed"):
        render_conversation(tok, conv)


def test_conversation_starting_with_assistant_raises(tok: Tokenizer) -> None:
    conv = {"messages": [{"role": "assistant", "content": "hi"}]}  # type: ignore[typeddict-item]
    with pytest.raises(ValueError, match="must alternate"):
        render_conversation(tok, conv)


def test_consecutive_same_role_raises(tok: Tokenizer) -> None:
    conv: Conversation = {
        "messages": [
            {"role": "user", "content": "ping"},
            {"role": "user", "content": "ping again"},
        ],
    }
    with pytest.raises(ValueError, match="must alternate"):
        render_conversation(tok, conv)


def test_empty_conversation_raises(tok: Tokenizer) -> None:
    conv: Conversation = {"messages": []}
    with pytest.raises(ValueError, match="no messages"):
        render_conversation(tok, conv)


def test_render_for_completion_merges_system_message(tok: Tokenizer) -> None:
    """render_for_completion applies the same system-merge as render_conversation."""
    conv: Conversation = {
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ],
    }
    ids = render_for_completion(tok, conv)
    assert ids[-1] == tok.encode_special(SpecialToken.ASSISTANT_START)
    # Decode everything except the final primer token so the merged prefix
    # can be round-tripped through the BPE without the special token.
    decoded = tok.decode(ids[:-1])
    assert "be helpful\n\nhi" in decoded


def test_render_for_completion_rejects_bad_alternation(tok: Tokenizer) -> None:
    conv = {"messages": [{"role": "assistant", "content": "hi"}]}  # type: ignore[typeddict-item]
    with pytest.raises(ValueError, match="must alternate"):
        render_for_completion(tok, conv)


def test_render_for_completion_ends_with_assistant_start(tok: Tokenizer) -> None:
    conv: Conversation = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    ids = render_for_completion(tok, conv)
    assert ids[-1] == tok.encode_special(SpecialToken.ASSISTANT_START)


def test_render_for_completion_strips_last_assistant(tok: Tokenizer) -> None:
    conv: Conversation = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    ids = render_for_completion(tok, conv)
    decoded = tok.decode(ids)
    assert "hello" not in decoded
    assert "hi" in decoded
