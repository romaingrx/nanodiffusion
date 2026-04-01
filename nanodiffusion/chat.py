"""Chat formatting: conversation rendering with loss masks."""

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar, Literal, TypedDict

from nanodiffusion.tokenizer import SpecialToken, Tokenizer

Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class Conversation(TypedDict):
    messages: list[Message]


@dataclasses.dataclass
class SequenceBuilder:
    """Builds a token sequence with parallel loss masks.

    Uses context managers to scope start/end delimiter pairs and
    automatically set the supervised flag for content within a turn.
    """

    _tok: Tokenizer
    ids: list[int] = dataclasses.field(default_factory=list)
    mask: list[int] = dataclasses.field(default_factory=list)
    _supervised: bool = False

    def special(self, token: SpecialToken) -> None:
        self.ids.append(self._tok.encode_special(token))
        self.mask.append(int(self._supervised))

    def text(self, content: str) -> None:
        tokens = self._tok.encode(content)
        self.ids.extend(tokens)
        self.mask.extend([int(self._supervised)] * len(tokens))

    @contextmanager
    def turn(
        self,
        start: SpecialToken,
        end: SpecialToken,
        *,
        supervised: bool = False,
    ) -> Iterator[None]:
        """Scope a turn between *start* and *end* delimiters.

        The start delimiter inherits the outer supervised state (typically 0).
        Content and the end delimiter use *supervised*.
        """
        self.special(start)
        prev, self._supervised = self._supervised, supervised
        try:
            yield
        finally:
            self.special(end)
            self._supervised = prev

    _ROLE_DELIMITERS: ClassVar[dict[Role, tuple[SpecialToken, SpecialToken, bool]]] = {
        "user": (SpecialToken.USER_START, SpecialToken.USER_END, False),
        "assistant": (SpecialToken.ASSISTANT_START, SpecialToken.ASSISTANT_END, True),
    }

    def message(self, msg: Message) -> None:
        role = msg["role"]
        entry = self._ROLE_DELIMITERS.get(role)
        if entry is None:
            err = f"Unknown role: {role!r}"
            raise ValueError(err)
        start, end, supervised = entry
        with self.turn(start, end, supervised=supervised):
            self.text(msg["content"])


def render_conversation(
    tok: Tokenizer,
    conversation: Conversation,
) -> tuple[list[int], list[int]]:
    """Tokenize a chat conversation with loss masks for training.

    Only assistant content has ``mask=1`` (supervised); everything else
    is ``mask=0`` (context).
    """
    b = SequenceBuilder(tok)
    b.special(SpecialToken.BOS)
    for msg in conversation["messages"]:
        b.message(msg)
    b.special(SpecialToken.EOS)
    return b.ids, b.mask


def render_for_completion(
    tok: Tokenizer,
    conversation: Conversation,
) -> list[int]:
    """Render a conversation primed for assistant completion.

    Drops the final assistant message (if present) and appends
    ``<|assistant_start|>`` so the model can continue generation.
    """
    messages = conversation["messages"]
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]

    ids, _ = render_conversation(tok, {"messages": messages})
    # render_conversation always ends with EOS; replace it to prime generation
    ids[-1] = tok.encode_special(SpecialToken.ASSISTANT_START)
    return ids
