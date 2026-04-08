"""Chat formatting: conversation rendering with loss masks."""

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar, Literal, TypedDict

from nanodiffusion.tokenizer import SpecialToken, Tokenizer

Role = Literal["user", "assistant", "system"]
SupervisedRole = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class Conversation(TypedDict):
    messages: list[Message]


def _merge_system_message(messages: list[Message]) -> list[Message]:
    """Prepend an optional leading system message to the first user message.

    Mirrors ``nanochat/tokenizer.py::render_conversation``. The concatenation
    uses ``\\n\\n`` so the combined text reads as two paragraphs. All other
    messages are returned unchanged, and the original list is never mutated.
    """
    if not messages or messages[0]["role"] != "system":
        return messages
    # Need at least the system + one user message for the merge to make sense.
    if len(messages) < 2 or messages[1]["role"] != "user":  # noqa: PLR2004
        err = "System message must be followed by a user message"
        raise ValueError(err)
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    merged_user: Message = {
        "role": "user",
        "content": f"{system_content}\n\n{user_content}",
    }
    return [merged_user, *messages[2:]]


def _validate_alternation(messages: list[Message]) -> None:
    """Require strict user/assistant alternation starting with ``user``.

    ``system`` must have been merged before this runs; hitting one here is a
    schema error, not silently accepted. Empty conversations also raise.
    """
    if not messages:
        err = "Conversation has no messages after system merge"
        raise ValueError(err)
    for i, msg in enumerate(messages):
        expected: SupervisedRole = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected:
            err = (
                f"Message {i} has role {msg['role']!r} but must alternate "
                f"user/assistant starting with 'user' (expected {expected!r})"
            )
            raise ValueError(err)


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

    _ROLE_DELIMITERS: ClassVar[
        dict[SupervisedRole, tuple[SpecialToken, SpecialToken, bool]]
    ] = {
        "user": (SpecialToken.USER_START, SpecialToken.USER_END, False),
        "assistant": (SpecialToken.ASSISTANT_START, SpecialToken.ASSISTANT_END, True),
    }

    def message(self, msg: Message) -> None:
        role = msg["role"]
        if role not in self._ROLE_DELIMITERS:
            err = (
                f"Unknown role: {role!r}. System messages must be merged into "
                "the first user message before reaching SequenceBuilder."
            )
            raise ValueError(err)
        start, end, supervised = self._ROLE_DELIMITERS[role]
        with self.turn(start, end, supervised=supervised):
            self.text(msg["content"])


def render_conversation(
    tok: Tokenizer,
    conversation: Conversation,
) -> tuple[list[int], list[int]]:
    """Tokenize a chat conversation with loss masks for training.

    Only assistant content has ``mask=1`` (supervised); everything else
    is ``mask=0`` (context). An optional leading ``system`` message is
    merged into the first user message (``\\n\\n`` separator) before
    rendering, matching nanochat's convention. The resulting messages
    must strictly alternate user/assistant starting with user — violations
    raise :class:`ValueError` at load time instead of producing silently
    wrong supervision masks.
    """
    messages = _merge_system_message(conversation["messages"])
    _validate_alternation(messages)
    b = SequenceBuilder(tok)
    b.special(SpecialToken.BOS)
    for msg in messages:
        b.message(msg)
    b.special(SpecialToken.EOS)
    return b.ids, b.mask


def render_for_completion(
    tok: Tokenizer,
    conversation: Conversation,
) -> list[int]:
    """Render a conversation primed for assistant completion.

    Applies the same system-merge + alternation checks as
    :func:`render_conversation`, drops the final assistant message (if
    present), and appends ``<|assistant_start|>`` so the model can
    continue generation.
    """
    messages = _merge_system_message(conversation["messages"])
    _validate_alternation(messages)
    if messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    # Recurse through render_conversation for the BOS/turn rendering, but
    # bypass its validators: we already ran them on the pre-drop sequence,
    # and a single trailing ``user`` here is expected after the drop.
    b = SequenceBuilder(tok)
    b.special(SpecialToken.BOS)
    for msg in messages:
        b.message(msg)
    b.special(SpecialToken.EOS)
    ids = b.ids
    # SequenceBuilder always ends with EOS; replace it to prime generation.
    ids[-1] = tok.encode_special(SpecialToken.ASSISTANT_START)
    return ids
