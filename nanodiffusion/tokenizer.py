"""Thin tiktoken wrapper with MASK and chat special tokens.

Follows the nanochat convention of paired start/end delimiters and
``<|...|>`` naming for special tokens.
"""

import enum
from typing import Protocol, runtime_checkable

import tiktoken


@runtime_checkable
class TokenizerLike(Protocol):
    """Minimal tokenizer surface the data loader depends on."""

    eos_token_id: int

    def encode_batch(
        self,
        texts: list[str],
        *,
        num_threads: int = 4,
    ) -> list[list[int]]: ...


class SpecialToken(enum.StrEnum):
    """Special tokens; IDs assigned in declaration order from ``base_vocab_size``."""

    MASK = "<|mask|>"
    BOS = "<|bos|>"
    EOS = "<|eos|>"
    USER_START = "<|user_start|>"
    USER_END = "<|user_end|>"
    ASSISTANT_START = "<|assistant_start|>"
    ASSISTANT_END = "<|assistant_end|>"


class Tokenizer:
    """GPT-2 tiktoken encoding extended with diffusion and chat tokens."""

    def __init__(self) -> None:
        self._base = tiktoken.get_encoding("gpt2")
        base = self._base.n_vocab

        self._special_to_id: dict[SpecialToken, int] = {
            tok: base + i for i, tok in enumerate(SpecialToken)
        }
        self._id_to_special: dict[int, SpecialToken] = {
            v: k for k, v in self._special_to_id.items()
        }

        self.base_vocab_size: int = base
        self.vocab_size: int = base + len(self._special_to_id)
        self.mask_token_id: int = self._special_to_id[SpecialToken.MASK]
        self.bos_token_id: int = self._special_to_id[SpecialToken.BOS]
        self.eos_token_id: int = self._special_to_id[SpecialToken.EOS]

    def encode_special(self, token: SpecialToken) -> int:
        return self._special_to_id[token]

    def encode(self, text: str) -> list[int]:
        return self._base.encode(text)

    def encode_batch(
        self,
        texts: list[str],
        *,
        num_threads: int = 4,
    ) -> list[list[int]]:
        """Encode plain text (no special tokens). GIL-free via tiktoken C++."""
        return self._base.encode_ordinary_batch(texts, num_threads=num_threads)

    def decode(self, token_ids: list[int]) -> str:
        parts: list[str] = []
        buf: list[int] = []

        for tid in token_ids:
            if tid in self._id_to_special:
                if buf:
                    parts.append(self._base.decode(buf))
                    buf = []
                parts.append(self._id_to_special[tid])
            else:
                buf.append(tid)

        if buf:
            parts.append(self._base.decode(buf))

        return "".join(parts)
