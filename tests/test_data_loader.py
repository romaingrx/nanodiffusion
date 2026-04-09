import itertools
import threading
import time
from collections.abc import Iterator
from itertools import islice

import numpy as np
import pytest
from jaxtyping import TypeCheckError

from nanodiffusion.data.cursors import PretrainCursor
from nanodiffusion.data.loader import (
    BatchOutput,
    PrefetchIterator,
    prefetch,
    pretrain_loader,
)
from nanodiffusion.data.source import InMemoryTextSource, Split
from nanodiffusion.tokenizer import Tokenizer

_ShapeError = TypeCheckError


def _docs(n: int, words_per_doc: int = 20) -> list[str]:
    return [f"doc {i} " + ("hello world " * words_per_doc) for i in range(n)]


def test_batch_shape_and_dtype(tok: Tokenizer) -> None:
    src = InMemoryTextSource(_docs(50), val_size=2)
    loader = pretrain_loader(src, tok, batch_size=4, seq_len=32, split="train")
    batch = next(loader)

    assert batch.tokens.shape == (4, 32)
    assert batch.segments.shape == (4, 32)
    assert batch.tokens.dtype == np.int32
    assert batch.segments.dtype == np.int32


def test_segments_start_at_zero(tok: Tokenizer) -> None:
    src = InMemoryTextSource(_docs(50), val_size=2)
    loader = pretrain_loader(src, tok, batch_size=4, seq_len=32, split="train")
    batches = list(islice(loader, 3))
    for b in batches:
        assert (b.segments[:, 0] == 0).all()


def test_segments_non_negative_per_row(tok: Tokenizer) -> None:
    """The per-row min-subtraction must never produce negative segment ids."""
    src = InMemoryTextSource(_docs(80, words_per_doc=5), val_size=4)
    loader = pretrain_loader(src, tok, batch_size=4, seq_len=64, split="train")
    for batch in islice(loader, 4):
        assert (batch.segments >= 0).all()


def test_segments_monotonic_within_row(tok: Tokenizer) -> None:
    """Segment ids never decrease within a row and only increment by 1 at most."""
    src = InMemoryTextSource(_docs(80, words_per_doc=5), val_size=4)
    loader = pretrain_loader(src, tok, batch_size=4, seq_len=64, split="train")
    batches = list(islice(loader, 4))
    saw_increment = False
    for b in batches:
        diffs = np.diff(b.segments, axis=1)
        assert (diffs >= 0).all()
        assert (diffs <= 1).all()
        if (diffs == 1).any():
            saw_increment = True
    # Sanity: the loader actually produced at least one segment boundary
    # across the sample, otherwise the monotonicity check is vacuous.
    assert saw_increment


def test_segment_increments_after_eos(tok: Tokenizer) -> None:
    """Where the row contains an EOS, segment id increments at the next position."""
    src = InMemoryTextSource(_docs(80, words_per_doc=5), val_size=4)
    loader = pretrain_loader(src, tok, batch_size=2, seq_len=64, split="train")
    batch = next(loader)
    eos = tok.eos_token_id

    seen_eos = 0
    for row in range(2):
        eos_positions = np.flatnonzero(batch.tokens[row] == eos)
        seen_eos += int(eos_positions.size)
        for pos in eos_positions:
            if pos + 1 < batch.segments.shape[1]:
                assert batch.segments[row, pos + 1] == batch.segments[row, pos] + 1
    # The test only proves anything if at least one EOS appeared.
    assert seen_eos > 0


def test_full_utilization_no_special_tokens_in_text(tok: Tokenizer) -> None:
    """Greedy concat should not introduce padding or mask tokens into x0."""
    src = InMemoryTextSource(_docs(50), val_size=2)
    loader = pretrain_loader(src, tok, batch_size=4, seq_len=32, split="train")
    batch = next(loader)
    # mask token only appears via the diffusion forward pass
    assert (batch.tokens != tok.mask_token_id).all()
    # bos token is reserved for chat conversations, never in pretrain rows
    assert (batch.tokens != tok.bos_token_id).all()


def test_determinism(tok: Tokenizer) -> None:
    """Two runs with identical sources and parameters yield identical batches."""
    docs = _docs(60)

    def loader_run() -> list[BatchOutput]:
        src = InMemoryTextSource(docs, val_size=2)
        return list(
            islice(
                pretrain_loader(src, tok, batch_size=2, seq_len=48, split="train"),
                5,
            )
        )

    a = loader_run()
    b = loader_run()
    for ba, bb in zip(a, b, strict=True):
        np.testing.assert_array_equal(ba.tokens, bb.tokens)
        np.testing.assert_array_equal(ba.segments, bb.segments)
        assert ba.state == bb.state


def test_long_doc_spans_multiple_chunks(tok: Tokenizer) -> None:
    """A doc whose token count exceeds chunk_size must span several batches."""
    # Build a doc large enough to fill several chunks (chunk_size = 2*32 = 64).
    huge_doc = "lorem ipsum dolor sit amet consectetur adipiscing elit " * 200
    src = InMemoryTextSource([huge_doc, "tiny"], val_size=1)
    loader = pretrain_loader(src, tok, batch_size=2, seq_len=32, split="train")
    batches = list(islice(loader, 6))

    # Per-row reset still applies for every row even mid-doc.
    for b in batches:
        assert (b.segments[:, 0] == 0).all()
    # Most rows are still inside the long doc, so they have very few unique
    # segment ids. At least some rows should have only segment 0 (no EOS yet).
    assert any((b.segments == 0).all() for b in batches)


def _state_tuple(state: PretrainCursor) -> tuple[int, int, int]:
    return state.epoch, state.shard_idx, state.row_group_idx


def test_resume_state_advances_past_saved_position(tok: Tokenizer) -> None:
    """Resuming must skip past the saved position rather than restart."""
    src = InMemoryTextSource(_docs(80), val_size=4)
    first = list(
        islice(
            pretrain_loader(src, tok, batch_size=2, seq_len=32, split="train"),
            3,
        )
    )
    saved_state = first[-1].state

    resumed_src = InMemoryTextSource(_docs(80), val_size=4)
    resumed = list(
        islice(
            pretrain_loader(
                resumed_src,
                tok,
                batch_size=2,
                seq_len=32,
                split="train",
                resume_state=saved_state,
            ),
            2,
        )
    )

    # The first resumed batch must come from a strictly later source position;
    # otherwise resume_state is silently ignored.
    assert _state_tuple(resumed[0].state) > _state_tuple(saved_state)
    for b in resumed:
        assert b.tokens.shape == (2, 32)
        assert (b.segments[:, 0] == 0).all()


def test_state_epoch_monotonic(tok: Tokenizer) -> None:
    src = InMemoryTextSource(_docs(80), val_size=4)
    loader = pretrain_loader(src, tok, batch_size=2, seq_len=32, split="train")
    batches = list(islice(loader, 5))
    epochs = [b.state.epoch for b in batches]
    assert epochs[0] >= 1
    for a, b in itertools.pairwise(epochs):
        assert b >= a


def test_pretrain_loader_rejects_invalid_dims(tok: Tokenizer) -> None:
    src = InMemoryTextSource(_docs(10), val_size=1)
    with pytest.raises(ValueError, match="positive"):
        next(pretrain_loader(src, tok, batch_size=0, seq_len=32, split="train"))
    with pytest.raises(ValueError, match="positive"):
        next(pretrain_loader(src, tok, batch_size=2, seq_len=0, split="train"))


def test_pretrain_loader_infinite_loop_guard(tok: Tokenizer) -> None:
    """A tokenizer that always returns empty must not loop forever."""

    class EmptyTokenizer:
        eos_token_id: int = 0

        def encode_batch(self, texts: list[str]) -> list[list[int]]:
            return [[] for _ in texts]

    del tok
    src = InMemoryTextSource(_docs(50), val_size=2)
    with pytest.raises(RuntimeError, match="no tokens"):
        next(
            pretrain_loader(
                src,
                EmptyTokenizer(),
                batch_size=2,
                seq_len=32,
                split="train",
                max_empty_passes=5,
            )
        )


def test_pretrain_loader_handles_finite_source(tok: Tokenizer) -> None:
    """A finite source must surface as a clean StopIteration, not RuntimeError."""
    from structlog.testing import capture_logs  # noqa: PLC0415

    class FiniteSource:
        def iter_documents(
            self,
            split: Split,
            *,
            start: int = 0,
            step: int = 1,
            batch_size: int = 128,
            resume: PretrainCursor | None = None,
        ) -> Iterator[tuple[list[str], PretrainCursor]]:
            del split, start, step, batch_size, resume
            position = PretrainCursor(epoch=1, shard_idx=0, row_group_idx=0)
            yield ["just one doc"], position

    loader = pretrain_loader(
        FiniteSource(),
        tok,
        batch_size=4,
        seq_len=64,
        split="train",
    )
    # The single tiny doc cannot fill a 256-token chunk; the loader should
    # exhaust the source, log a warning, and exit cleanly.
    with capture_logs() as logs:
        result = list(loader)
    assert result == []
    assert any("partial chunk dropped" in log["event"] for log in logs)


def test_batch_output_to_jax_returns_jax_batch(tok: Tokenizer) -> None:
    import jax  # noqa: PLC0415

    from nanodiffusion.data.loader import JaxBatch  # noqa: PLC0415

    src = InMemoryTextSource(_docs(40), val_size=2)
    batch = next(pretrain_loader(src, tok, batch_size=2, seq_len=16, split="train"))
    jax_batch = batch.to_jax()
    assert isinstance(jax_batch, JaxBatch)
    assert isinstance(jax_batch.tokens, jax.Array)
    assert isinstance(jax_batch.segments, jax.Array)
    assert jax_batch.tokens.shape == (2, 16)
    assert jax_batch.segments.shape == (2, 16)
    # state must round-trip unchanged so resume code can keep using it.
    assert jax_batch.state == batch.state


def test_batch_output_validates_shape_mismatch() -> None:
    state = PretrainCursor(epoch=1, shard_idx=0, row_group_idx=0)
    with pytest.raises(_ShapeError):
        BatchOutput(
            tokens=np.zeros((4, 8), dtype=np.int32),
            segments=np.zeros((4, 7), dtype=np.int32),
            state=state,
        )


def test_batch_output_validates_ndim() -> None:
    state = PretrainCursor(epoch=1, shard_idx=0, row_group_idx=0)
    with pytest.raises(_ShapeError):
        BatchOutput(
            tokens=np.zeros((8,), dtype=np.int32),
            segments=np.zeros((8,), dtype=np.int32),
            state=state,
        )


def test_batch_output_validates_dtype() -> None:
    """jaxtyping rejects float arrays for Int[...] annotations."""
    state = PretrainCursor(epoch=1, shard_idx=0, row_group_idx=0)
    with pytest.raises(_ShapeError):
        BatchOutput(
            tokens=np.zeros((4, 8), dtype=np.float32),
            segments=np.zeros((4, 8), dtype=np.int32),
            state=state,
        )


def test_prefetch_yields_same_sequence_as_loader(tok: Tokenizer) -> None:
    src_a = InMemoryTextSource(_docs(60), val_size=2)
    src_b = InMemoryTextSource(_docs(60), val_size=2)

    plain = list(
        islice(
            pretrain_loader(src_a, tok, batch_size=2, seq_len=32, split="train"),
            4,
        )
    )
    with prefetch(
        pretrain_loader(src_b, tok, batch_size=2, seq_len=32, split="train"),
        size=2,
    ) as p:
        prefetched = list(islice(p, 4))

    for x, y in zip(plain, prefetched, strict=True):
        np.testing.assert_array_equal(x.tokens, y.tokens)
        np.testing.assert_array_equal(x.segments, y.segments)


def test_prefetch_close_is_idempotent() -> None:
    def gen() -> Iterator[int]:
        yield from range(100)

    p = PrefetchIterator(gen(), size=2)
    next(p)
    p.close()
    p.close()  # second call must not raise


def test_prefetch_next_after_close_raises_stop_iteration() -> None:
    """Regression: previously close() ate the sentinel and next() hung."""

    def gen() -> Iterator[int]:
        yield from range(100)

    p = PrefetchIterator(gen(), size=2)
    next(p)
    p.close()
    with pytest.raises(StopIteration):
        next(p)


def test_prefetch_close_returns_promptly() -> None:
    """close() must return without blocking the consumer indefinitely.

    With the executor-based implementation there is no "stuck on a full
    queue" failure mode (the executor handles backpressure for us), but we
    still want to verify that an unconsumed iterator can be torn down
    without leaving the executor running.
    """
    started = threading.Event()

    def fast_gen() -> Iterator[int]:
        started.set()
        yield from range(1000)

    p = PrefetchIterator(fast_gen(), size=4)
    assert started.wait(timeout=1.0)
    # Let the executor fill its look-ahead window.
    time.sleep(0.05)
    t0 = time.monotonic()
    p.close()
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5, f"close() took {elapsed:.2f}s"
    # After close(), next() must raise StopIteration cleanly.
    with pytest.raises(StopIteration):
        next(p)


def test_prefetch_propagates_worker_exceptions() -> None:
    class BoomError(Exception):
        pass

    def explode() -> Iterator[int]:
        yield 1
        raise BoomError("kaboom")

    p = PrefetchIterator(explode(), size=2)
    assert next(p) == 1
    with pytest.raises(BoomError, match="kaboom"):
        next(p)
    p.close()


def test_prefetch_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        PrefetchIterator(iter([1, 2, 3]), size=0)


def test_prefetch_runs_in_background_thread() -> None:
    """The worker thread should fill the queue without main-thread blocking.

    Uses an Event for synchronization (no sleeps) to avoid CI flakiness.
    """
    item_produced = threading.Event()
    second_item_produced = threading.Event()
    produced: list[int] = []

    def gen() -> Iterator[int]:
        for i in range(5):
            produced.append(i)
            if i == 0:
                item_produced.set()
            elif i == 1:
                second_item_produced.set()
            yield i

    with prefetch(gen(), size=4) as p:
        # The worker should produce at least the first two items eagerly,
        # without the main thread reading from the queue at all.
        assert item_produced.wait(timeout=1.0)
        assert second_item_produced.wait(timeout=1.0)
        items = list(p)
    assert items == [0, 1, 2, 3, 4]
