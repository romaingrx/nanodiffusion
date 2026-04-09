"""GSM8K grade-school math reasoning dataset.

Gold answers interleave prose with ``<<expr=result>>`` calculator
markers. We flatten the markers to plain text; the surrounding prose
already contains both the expression and the result
(e.g. ``"12/60 = $0.2 per minute"``), so stripping preserves answer
correctness without a tokenizer vocabulary change. Proper multi-part
tool-call support is deferred.
"""

import re

from nanodiffusion.chat import Conversation
from nanodiffusion.data.chat_datasets._base import register_hf_chat

# Require ``=`` inside the delimiters so literal ``<<`` / ``>>`` in
# natural text isn't silently consumed by the flattening regex.
_TOOL_RE = re.compile(r"<<[^<>]*=[^<>]*>>")


def strip_tool_calls(answer: str) -> str:
    return _TOOL_RE.sub("", answer)


def row_to_conversation(row: dict[str, object]) -> Conversation:
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        err = f"GSM8K row has bad question/answer types: {row!r}"
        raise TypeError(err)
    cleaned = strip_tool_calls(answer)
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": cleaned},
        ],
    }


register_hf_chat(
    name="gsm8k",
    repo_id="openai/gsm8k",
    subset="main",
    split="train",
    row_to_conversation=row_to_conversation,
    doc="GSM8K (openai/gsm8k main). Tool-call markers flattened to plain text.",
)
