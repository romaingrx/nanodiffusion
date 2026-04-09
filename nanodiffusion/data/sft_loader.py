"""Pad-per-row SFT data loader.

Consumes any :class:`ChatSource` and emits ``(batch, seq_len)`` batches
carrying a parallel boolean ``loss_mask`` per position. Conversations
are rendered once via :func:`nanodiffusion.chat.render_conversation`
(which already merges optional system messages and asserts role
alternation), right-padded with EOS if they fit, or skipped if they
don't. The loader is stateless across epochs modulo a seeded shuffle
of the source's index space, so resuming from a saved
:class:`SFTCursor` gives a deterministic continuation of the
iteration order.
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

    ``eqx.Module`` registration means this class is a proper JAX pytree
    — equinox's JIT treats the fields as leaves and hashes the static
    structure, avoiding the ``unhashable ArrayImpl`` error a plain
    ``@dataclass(frozen=True)`` would hit inside ``equinox/_jit.py``.
    """

    tokens: TokenBatch
    loss_mask: MaskBatch


@dataclass(frozen=True, slots=True)
class SFTBatchOutput:
    """Host-side SFT batch: numpy arrays plus resume cursor.

    Mirrors :class:`nanodiffusion.data.loader.BatchOutput`'s split so
    the cursor never enters JIT and ``prefetch`` can still wrap it.
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

    Keying the RNG off an ``(seed, epoch)`` mix means two runs with the
    same seed produce identical iteration orders, and resume-from-cursor
    reproduces the second half of an interrupted epoch exactly. We
    combine seed and epoch via a cheap 64-bit hash instead of passing a
    tuple — ``random.Random`` accepts ints but the stubs don't allow
    tuples.
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

    Each epoch is a fresh seeded shuffle of the source index space.
    Conversations that don't fit in ``seq_len`` after rendering are
    skipped (not truncated) — nanochat's approach, and safer than
    slicing off chat delimiters that the model never saw at pretrain.
    Short conversations are right-padded with EOS (``loss_mask=0`` on
    pad), since EOS already means "document boundary" to the pretrained
    model.

    ``resume_state`` fast-forwards into the middle of an epoch so
    resumption is deterministic given the same seed.

    ``max_empty_passes`` bounds the number of consecutive skipped
    conversations (too long or no supervised tokens) before the loader
    aborts — same safety net as the pretrain loader against
    silently-empty data.
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
