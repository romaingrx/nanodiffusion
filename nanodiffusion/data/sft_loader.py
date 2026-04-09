"""Pad-per-row SFT data loader.

Consumes any :class:`ChatSource` and emits ``(batch, seq_len)``
batches with a parallel boolean ``loss_mask``. Conversations are
rendered via :func:`nanodiffusion.chat.render_conversation`, right-
padded with EOS if they fit, or skipped if they don't. Iteration
order is a seeded per-epoch shuffle so saved :class:`SFTCursor`
resumes pick up where they left off.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from nanodiffusion.chat import render_conversation
from nanodiffusion.data.chat_source import ChatSource
from nanodiffusion.data.cursors import SFTCursor
from nanodiffusion.tokenizer import Tokenizer
from nanodiffusion.types import (
    MaskBatch,
    NumpyLossMaskBatch,
    NumpyTokenBatch,
    TokenBatch,
)


class SFTJaxBatch(eqx.Module):
    """JIT-visible pytree: tokens + loss_mask only, no host-side cursor.

    Inheriting from :class:`eqx.Module` registers this as a proper JAX
    pytree — a plain ``@dataclass(frozen=True)`` would hit an
    ``unhashable ArrayImpl`` error inside ``equinox/_jit.py``.
    """

    tokens: TokenBatch
    loss_mask: MaskBatch


@dataclass(frozen=True, slots=True)
class SFTBatchOutput:
    """Host-side SFT batch: numpy arrays plus resume cursor.

    The cursor is kept on this host-side wrapper so it never enters
    JIT; :meth:`to_jax` returns the pytree-only :class:`SFTJaxBatch`.
    """

    tokens: NumpyTokenBatch
    loss_mask: NumpyLossMaskBatch
    state: SFTCursor

    def to_jax(self) -> SFTJaxBatch:
        return SFTJaxBatch(
            tokens=jnp.asarray(self.tokens),
            loss_mask=jnp.asarray(self.loss_mask),
        )


def _epoch_permutation(seed: int, epoch: int, n: int) -> list[int]:
    """Deterministic per-epoch shuffle of ``range(n)``.

    The ``(seed, epoch)`` pair is mixed via Knuth's multiplicative
    hash into a single 64-bit int so two runs with the same seed
    produce identical iteration orders and resume reproduces the
    second half of an interrupted epoch exactly.
    """
    combined = (seed * 2654435761 + epoch) & 0xFFFFFFFFFFFFFFFF
    rng = random.Random(combined)  # noqa: S311  # deterministic, not crypto
    order = list(range(n))
    rng.shuffle(order)
    return order


def _pad_row(
    ids: list[int],
    mask: list[int],
    *,
    seq_len: int,
    eos: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokens_row = np.full(seq_len, eos, dtype=np.int32)
    mask_row = np.zeros(seq_len, dtype=np.bool_)
    tokens_row[: len(ids)] = ids
    mask_row[: len(mask)] = np.array(mask, dtype=np.bool_)
    return tokens_row, mask_row


def sft_loader(
    source: ChatSource,
    tokenizer: Tokenizer,
    *,
    batch_size: int,
    seq_len: int,
    seed: int = 42,
    resume_state: SFTCursor | None = None,
    max_empty_passes: int = 100,
) -> Iterator[SFTBatchOutput]:
    """Yield SFT batches from ``source`` indefinitely.

    Oversized conversations are skipped (not truncated) — slicing off
    chat delimiters would feed the model a distribution pretrain
    never saw. Short conversations are EOS-padded with
    ``loss_mask=0`` on the pad, since EOS already means "document
    boundary" to the pretrained model. ``max_empty_passes`` bounds
    consecutive skips before the loader aborts, guarding against a
    source where nothing fits ``seq_len``.
    """
    if batch_size <= 0 or seq_len <= 0:
        msg = f"batch_size and seq_len must be positive, got {batch_size}, {seq_len}"
        raise ValueError(msg)
    n = len(source)
    if n == 0:
        err = "sft_loader source has zero length"
        raise ValueError(err)

    eos = tokenizer.eos_token_id
    epoch = resume_state.epoch if resume_state is not None else 1
    cursor = resume_state.permutation_idx + 1 if resume_state is not None else 0
    permutation = _epoch_permutation(seed, epoch, n)
    consecutive_skips = 0

    while True:
        rows_tokens: list[np.ndarray] = []
        rows_mask: list[np.ndarray] = []
        last_idx_in_epoch = cursor - 1

        while len(rows_tokens) < batch_size:
            if cursor >= n:
                epoch += 1
                permutation = _epoch_permutation(seed, epoch, n)
                cursor = 0

            source_idx = permutation[cursor]
            last_idx_in_epoch = cursor
            cursor += 1

            ids, mask = render_conversation(tokenizer, source[source_idx])
            if len(ids) > seq_len or not any(mask):
                consecutive_skips += 1
                if consecutive_skips >= max_empty_passes:
                    msg = (
                        f"sft_loader skipped {consecutive_skips} consecutive "
                        f"conversations (seq_len={seq_len}); aborting to avoid "
                        "an infinite skip loop"
                    )
                    raise RuntimeError(msg)
                continue
            consecutive_skips = 0

            tokens_row, mask_row = _pad_row(ids, mask, seq_len=seq_len, eos=eos)
            rows_tokens.append(tokens_row)
            rows_mask.append(mask_row)

        yield SFTBatchOutput(
            tokens=np.stack(rows_tokens),
            loss_mask=np.stack(rows_mask),
            state=SFTCursor(epoch=epoch, permutation_idx=last_idx_in_epoch),
        )
