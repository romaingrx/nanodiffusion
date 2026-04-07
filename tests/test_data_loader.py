import itertools
import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

from nanodiffusion.data.loader import (
    BatchOutput,
    PrefetchIterator,
    prefetch,
    pretrain_loader,
)
from nanodiffusion.data.source import InMemoryTextSource
from nanodiffusion.tokenizer import Tokenizer


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer()


def _docs(n: int, words_per_doc: int = 20) -> list[str]:
    return [f"doc {i} " + ("hello world " * words_per_doc) for i in range(n)]


def _take(it: Iterator[BatchOutput], n: int) -> list[BatchOutput]:
    return [next(it) for _ in range(n)]


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
    batches = _take(loader, 3)
    for b in batches:
        assert (b.segments[:, 0] == 0).all()


def test_segments_monotonic_within_row(tok: Tokenizer) -> None:
    """Segment ids never decrease within a row and only increment by 1 at most."""
    src = InMemoryTextSource(_docs(80, words_per_doc=5), val_size=4)
    loader = pretrain_loader(src, tok, batch_size=4, seq_len=64, split="train")
    batches = _take(loader, 4)
    for b in batches:
        diffs = np.diff(b.segments, axis=1)
        assert (diffs >= 0).all()
        assert (diffs <= 1).all()


def test_segment_increments_after_eos(tok: Tokenizer) -> None:
    """Where the row contains an EOS, the segment id increments at the next position."""
    src = InMemoryTextSource(_docs(80, words_per_doc=5), val_size=4)
    loader = pretrain_loader(src, tok, batch_size=2, seq_len=64, split="train")
    batch = next(loader)
    eos = tok.eos_token_id
    for row in range(2):
        eos_positions = np.flatnonzero(batch.tokens[row] == eos)
        if eos_positions.size == 0:
            continue
        for pos in eos_positions:
            if pos + 1 < batch.segments.shape[1]:
                assert batch.segments[row, pos + 1] == batch.segments[row, pos] + 1


def test_full_utilization_no_padding_token(tok: Tokenizer) -> None:
    """We use greedy concat, so there's no pad token in the output."""
    src = InMemoryTextSource(_docs(50), val_size=2)
    loader = pretrain_loader(src, tok, batch_size=4, seq_len=32, split="train")
    batch = next(loader)
    # The mask token id should never appear in pretrain data; only model
    # forward-mask injects it. EOS is allowed.
    assert (batch.tokens != tok.mask_token_id).all()


def test_determinism(tok: Tokenizer) -> None:
    """Two runs with identical sources and parameters yield identical batches."""
    docs = _docs(60)

    def loader_run() -> list[BatchOutput]:
        src = InMemoryTextSource(docs, val_size=2)
        return _take(
            pretrain_loader(src, tok, batch_size=2, seq_len=48, split="train"),
            5,
        )

    a = loader_run()
    b = loader_run()
    for ba, bb in zip(a, b, strict=True):
        np.testing.assert_array_equal(ba.tokens, bb.tokens)
        np.testing.assert_array_equal(ba.segments, bb.segments)
        assert ba.state == bb.state


def test_long_doc_spans_multiple_rows(tok: Tokenizer) -> None:
    """A doc longer than seq_len carries over the residue across batches."""
    long_doc = "lorem ipsum dolor sit amet " * 200
    src = InMemoryTextSource([long_doc] * 5, val_size=1)
    loader = pretrain_loader(src, tok, batch_size=2, seq_len=32, split="train")
    batches = _take(loader, 4)
    # Long-doc residue means each new batch's first row's first segment is 0
    # but the per-row reset should still apply.
    for b in batches:
        assert (b.segments[:, 0] == 0).all()


def test_resume_state_advances(tok: Tokenizer) -> None:
    src = InMemoryTextSource(_docs(80), val_size=4)
    loader = pretrain_loader(src, tok, batch_size=2, seq_len=32, split="train")
    batches = _take(loader, 5)
    epochs = [b.state["epoch"] for b in batches]
    assert epochs[0] >= 1
    # Epochs are monotonically non-decreasing across batches.
    for a, b in itertools.pairwise(epochs):
        assert b >= a


def test_pretrain_loader_rejects_invalid_dims(tok: Tokenizer) -> None:
    src = InMemoryTextSource(_docs(10), val_size=1)
    with pytest.raises(ValueError, match="positive"):
        next(pretrain_loader(src, tok, batch_size=0, seq_len=32, split="train"))
    with pytest.raises(ValueError, match="positive"):
        next(pretrain_loader(src, tok, batch_size=2, seq_len=0, split="train"))


def test_prefetch_yields_same_sequence_as_loader(tok: Tokenizer) -> None:
    src_a = InMemoryTextSource(_docs(60), val_size=2)
    src_b = InMemoryTextSource(_docs(60), val_size=2)

    plain = _take(
        pretrain_loader(src_a, tok, batch_size=2, seq_len=32, split="train"),
        4,
    )
    with prefetch(
        pretrain_loader(src_b, tok, batch_size=2, seq_len=32, split="train"),
        size=2,
    ) as p:
        prefetched = _take(p, 4)

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
    """The worker thread should make progress without main-thread blocking."""

    produced: list[int] = []
    started = threading.Event()

    def slow_gen() -> Iterator[int]:
        started.set()
        for i in range(5):
            time.sleep(0.005)
            produced.append(i)
            yield i

    with prefetch(slow_gen(), size=4) as p:
        assert started.wait(timeout=1.0)
        # Give the worker a moment to fill the queue.
        time.sleep(0.1)
        # By now the worker should have produced multiple items eagerly.
        assert len(produced) >= 2
        items = list(p)
    assert items == [0, 1, 2, 3, 4]
