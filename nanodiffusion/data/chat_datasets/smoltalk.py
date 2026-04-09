"""SmolTalk chat dataset.

General multi-turn chat corpus from HuggingFaceTB. Rows ship with a
``messages`` column that may lead with a ``system`` turn; the row
decoder leaves it in place so
:func:`nanodiffusion.chat.render_conversation` can merge it into the
first user message at render time.
"""

from nanodiffusion.chat import Conversation
from nanodiffusion.data.chat_datasets._base import (
    normalize_message,
    register_hf_chat,
)


def row_to_conversation(row: dict[str, object]) -> Conversation:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        err = f"SmolTalk row missing 'messages' list: {row!r}"
        raise TypeError(err)
    messages = [normalize_message(m) for m in raw_messages]
    if len(messages) < 2:  # noqa: PLR2004  # need at least one turn
        err = f"SmolTalk row has fewer than 2 messages: {messages!r}"
        raise ValueError(err)
    return {"messages": messages}


register_hf_chat(
    name="smoltalk",
    repo_id="HuggingFaceTB/smol-smoltalk",
    subset=None,
    split="train",
    row_to_conversation=row_to_conversation,
    doc="smol-smoltalk (HuggingFaceTB). ~460K general multi-turn chats.",
)
