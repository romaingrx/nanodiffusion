"""Streaming pretraining data loader with greedy concat and segment IDs.

Each batch carries a parallel ``segments`` array of per-row document ids
that resets to 0 at the start of every row so a future intra-document
attention masking pass can drop in without changing the loader interface.
"""

from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from nanodiffusion.data.source import SourcePosition, Split, TextSource
from nanodiffusion.tokenizer import TokenizerLike
from nanodiffusion.types import (
    NumpySegmentBatch,
    NumpyTokenBatch,
    SegmentBatch,
    TokenBatch,
)


@dataclass(frozen=True, slots=True)
class JaxBatch:
    tokens: TokenBatch
    segments: SegmentBatch
    state: SourcePosition


@dataclass(frozen=True, slots=True)
class BatchOutput:
    tokens: NumpyTokenBatch
    segments: NumpySegmentBatch
    state: SourcePosition

    def to_jax(self) -> JaxBatch:
        return JaxBatch(
            tokens=jnp.asarray(self.tokens),
            segments=jnp.asarray(self.segments),
            state=self.state,
        )


def pretrain_loader(  # noqa: PLR0915
    source: TextSource,
    tokenizer: TokenizerLike,
    *,
    batch_size: int,
    seq_len: int,
    split: Split,
    tokenizer_batch_size: int = 128,
    tokenizer_threads: int = 4,
    start: int = 0,
    step: int = 1,
    resume_state: SourcePosition | None = None,
    max_empty_passes: int = 100,
) -> Iterator[BatchOutput]:
    """Greedy-concat pretrain loader with EOS document separators.

    Pulls document batches from ``source``, tokenizes them via
    ``TokenizerLike.encode_batch`` (multi-threaded, GIL-free for tiktoken),
    accumulates tokens with trailing EOS, and yields ``(batch_size, seq_len)``
    numpy chunks. Each yield's ``segments`` array is renumbered per row so
    the first segment of every row is 0.

    ``max_empty_passes`` guards against the silent infinite loop that would
    occur if the tokenizer returned empty for every input doc (e.g. wrong
    special-token config).
    """
    if batch_size <= 0 or seq_len <= 0:
        msg = f"batch_size and seq_len must be positive, got {batch_size}, {seq_len}"
        raise ValueError(msg)

    eos = tokenizer.eos_token_id
    chunk_size = batch_size * seq_len
    pending_tokens: list[np.ndarray] = []
    pending_segments: list[np.ndarray] = []
    pending_size = 0
    next_doc_id = 0
    empty_passes = 0
    last_state: SourcePosition = resume_state or {
        "epoch": 1,
        "shard_idx": 0,
        "row_group_idx": 0,
    }

    docs_iter = source.iter_documents(
        split,
        start=start,
        step=step,
        batch_size=tokenizer_batch_size,
    )

    while True:
        while pending_size < chunk_size:
            try:
                doc_batch, position = next(docs_iter)
            except StopIteration:
                # PEP 479: bare StopIteration would become RuntimeError here.
                return
            last_state = position
            encoded = tokenizer.encode_batch(doc_batch, num_threads=tokenizer_threads)
            produced = False
            for doc_tokens in encoded:
                if not doc_tokens:
                    continue
                produced = True
                n = len(doc_tokens) + 1  # +1 for trailing EOS
                tok_arr = np.empty(n, dtype=np.int32)
                tok_arr[:-1] = doc_tokens
                tok_arr[-1] = eos
                pending_tokens.append(tok_arr)
                pending_segments.append(np.full(n, next_doc_id, dtype=np.int32))
                pending_size += n
                next_doc_id += 1
            if produced:
                empty_passes = 0
            else:
                empty_passes += 1
                if empty_passes >= max_empty_passes:
                    msg = (
                        f"Tokenizer produced no tokens for {empty_passes} "
                        "consecutive source batches; aborting to avoid an "
                        "infinite buffer-fill loop"
                    )
                    raise RuntimeError(msg)

        all_tokens = np.concatenate(pending_tokens)
        all_segments = np.concatenate(pending_segments)
        tokens_arr = all_tokens[:chunk_size]
        segments_arr = all_segments[:chunk_size]
        tail_t = all_tokens[chunk_size:]
        tail_s = all_segments[chunk_size:]
        if tail_t.size:
            pending_tokens = [tail_t]
            pending_segments = [tail_s]
            pending_size = tail_t.size
        else:
            pending_tokens = []
            pending_segments = []
            pending_size = 0
            next_doc_id = 0  # buffer fully drained; counter stays int32-safe

        tokens_chunk = tokens_arr.reshape(batch_size, seq_len)
        segments_chunk = segments_arr.reshape(batch_size, seq_len)
        segments_chunk = segments_chunk - segments_chunk.min(axis=1, keepdims=True)

        yield BatchOutput(
            tokens=tokens_chunk,
            segments=segments_chunk,
            state=last_state,
        )


class PrefetchIterator[T]:
    """Background producer with a bounded look-ahead window.

    Runs ``next(source)`` on a single worker thread via a
    :class:`ThreadPoolExecutor` so the consumer is never blocked on CPU work
    while the GPU is busy. Exceptions from the source surface at
    :meth:`__next__` via ``Future.result()``; ``StopIteration`` is preserved
    for finite sources.
    """

    def __init__(self, source: Iterator[T], size: int) -> None:
        if size <= 0:
            msg = f"prefetch size must be positive, got {size}"
            raise ValueError(msg)
        self._source = source
        self._size = size
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="prefetch"
        )
        self._inflight: deque[Future[T]] = deque()
        self._fill()

    def _fill(self) -> None:
        while len(self._inflight) < self._size:
            self._inflight.append(self._executor.submit(next, self._source))

    def __iter__(self) -> "PrefetchIterator[T]":
        return self

    def __next__(self) -> T:
        if not self._inflight:
            raise StopIteration
        future = self._inflight.popleft()
        try:
            value = future.result()
        except StopIteration:
            self._inflight.clear()
            raise
        self._fill()
        return value

    def close(self) -> None:
        """Shut down the worker. Idempotent and safe to call from any thread."""
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._inflight.clear()

    def __enter__(self) -> "PrefetchIterator[T]":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def prefetch[T](it: Iterator[T], size: int = 4) -> PrefetchIterator[T]:
    return PrefetchIterator(it, size)
