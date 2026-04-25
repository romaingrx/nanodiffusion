"""Streaming pretraining data loader with greedy concat and segment IDs.

Each batch carries a parallel ``segments`` array of per-row document ids
that resets to 0 at the start of every row so a future intra-document
attention masking pass can drop in without changing the loader interface.
"""

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import structlog

from nanodiffusion.data.cursors import PretrainCursor
from nanodiffusion.data.source import Split, TextSource
from nanodiffusion.tokenizer import TokenizerLike
from nanodiffusion.types import (
    NumpySegmentBatch,
    NumpyTokenBatch,
    SegmentBatch,
    TokenBatch,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class JaxBatch:
    tokens: TokenBatch
    segments: SegmentBatch
    state: PretrainCursor


@dataclass(frozen=True, slots=True)
class BatchOutput:
    tokens: NumpyTokenBatch
    segments: NumpySegmentBatch
    state: PretrainCursor

    def to_jax(self) -> JaxBatch:
        return JaxBatch(
            tokens=jnp.asarray(self.tokens),
            segments=jnp.asarray(self.segments),
            state=self.state,
        )


@dataclass(frozen=True, slots=True)
class _TokenSpan:
    tokens: np.ndarray
    start: PretrainCursor
    total_doc_tokens: int
    segment_id: int

    def cursor_after(self, consumed: int) -> PretrainCursor:
        token_offset = self.start.token_offset + consumed
        if token_offset < self.total_doc_tokens:
            return self.start.model_copy(update={"token_offset": token_offset})
        return self.start.model_copy(
            update={"doc_idx": self.start.doc_idx + 1, "token_offset": 0}
        )

    def tail(self, consumed: int) -> "_TokenSpan":
        return _TokenSpan(
            tokens=self.tokens[consumed:],
            start=self.cursor_after(consumed),
            total_doc_tokens=self.total_doc_tokens,
            segment_id=self.segment_id,
        )


class _ChunkBuffer:
    """Mutable token + segment-id buffer used by :func:`pretrain_loader`.

    Each appended document gets a fresh, monotonically increasing segment id
    so that downstream code can recover document boundaries from the segments
    array. ``drain`` slices off exactly ``chunk_size`` tokens and keeps any
    overflow as the seed for the next chunk.
    """

    def __init__(self) -> None:
        self._spans: deque[_TokenSpan] = deque()
        self._size = 0
        self._next_doc_id = 0

    @property
    def size(self) -> int:
        return self._size

    def append_doc(
        self, doc_tokens: list[int], eos: int, cursor: PretrainCursor
    ) -> bool:
        total = len(doc_tokens) + 1
        if cursor.token_offset >= total:
            return False
        tok = np.empty(total, dtype=np.int32)
        tok[:-1] = doc_tokens
        tok[-1] = eos
        remaining = tok[cursor.token_offset :]
        self._spans.append(
            _TokenSpan(
                tokens=remaining,
                start=cursor,
                total_doc_tokens=total,
                segment_id=self._next_doc_id,
            )
        )
        self._size += remaining.size
        self._next_doc_id += 1
        return True

    def drain(self, chunk_size: int) -> tuple[np.ndarray, np.ndarray, PretrainCursor]:
        """Cut off ``chunk_size`` tokens. Caller must ensure size >= chunk_size."""
        remaining = chunk_size
        token_parts: list[np.ndarray] = []
        segment_parts: list[np.ndarray] = []
        next_cursor: PretrainCursor | None = None

        while remaining > 0:
            span = self._spans.popleft()
            take = min(remaining, span.tokens.size)
            token_parts.append(span.tokens[:take])
            segment_parts.append(np.full(take, span.segment_id, dtype=np.int32))
            next_cursor = span.cursor_after(take)
            self._size -= take
            remaining -= take
            if take < span.tokens.size:
                self._spans.appendleft(span.tail(take))

        if self._size == 0:
            self._next_doc_id = 0
        if next_cursor is None:
            msg = "cannot drain an empty chunk buffer"
            raise RuntimeError(msg)
        return np.concatenate(token_parts), np.concatenate(segment_parts), next_cursor


def pretrain_loader(
    source: TextSource,
    tokenizer: TokenizerLike,
    *,
    batch_size: int,
    seq_len: int,
    split: Split,
    tokenizer_batch_size: int = 128,
    start: int = 0,
    step: int = 1,
    resume_state: PretrainCursor | None = None,
    max_empty_passes: int = 100,
) -> Iterator[BatchOutput]:
    """Greedy-concat pretrain loader with EOS document separators.

    Pulls document batches from ``source``, tokenizes them via
    ``TokenizerLike.encode_batch``, accumulates tokens with trailing EOS,
    and yields ``(batch_size, seq_len)`` numpy chunks. Each yield's
    ``segments`` array is renumbered per row so the first segment of every
    row is 0.

    Each yielded ``state`` is the exact cursor for the next token after
    the emitted batch, so checkpoint ``step_N`` resumes at precisely the
    first token for batch ``N + 1``. ``max_empty_passes`` guards against
    the silent infinite loop that would occur if the tokenizer returned
    empty for every input doc (e.g. wrong special-token config).
    """
    if batch_size <= 0 or seq_len <= 0:
        msg = f"batch_size and seq_len must be positive, got {batch_size}, {seq_len}"
        raise ValueError(msg)

    eos = tokenizer.eos_token_id
    chunk_size = batch_size * seq_len
    buffer = _ChunkBuffer()
    empty_passes = 0
    docs_iter = source.iter_documents(
        split,
        start=start,
        step=step,
        batch_size=tokenizer_batch_size,
        resume=resume_state,
    )

    while True:
        while buffer.size < chunk_size:
            try:
                doc_batch, position = next(docs_iter)
            except StopIteration:
                # PEP 479: bare StopIteration would become RuntimeError here.
                if buffer.size > 0:
                    logger.warning(
                        "source exhausted with pending tokens; partial chunk dropped",
                        pending_tokens=buffer.size,
                        chunk_size=chunk_size,
                    )
                return
            encoded = tokenizer.encode_batch(doc_batch)
            produced = False
            for local_idx, doc_tokens in enumerate(encoded):
                if not doc_tokens:
                    continue
                doc_cursor = position.cursor_for_batch_doc(local_idx)
                token_offset = (
                    0
                    if resume_state is None
                    else resume_state.token_offset_for(doc_cursor)
                )
                cursor = doc_cursor.with_token_offset(token_offset)
                produced = buffer.append_doc(doc_tokens, eos, cursor) or produced
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

        chunk_tokens, chunk_segments, next_cursor = buffer.drain(chunk_size)
        tokens_chunk = chunk_tokens.reshape(batch_size, seq_len)
        segments_chunk = chunk_segments.reshape(batch_size, seq_len)
        # Tokens within a row are written in document order, so the per-row
        # min equals the row's first segment id; subtracting it makes every
        # row start at 0 without changing intra-row boundaries.
        segments_chunk = segments_chunk - segments_chunk.min(axis=1, keepdims=True)

        yield BatchOutput(
            tokens=tokens_chunk,
            segments=segments_chunk,
            state=next_cursor,
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


class DevicePrefetchIterator[B, JB]:
    """Two-stage prefetch: CPU batches → async device_put → ready on device.

    Stage 1 (:class:`PrefetchIterator`) prepares numpy batches on a
    background thread. Stage 2 calls ``prepare_fn`` on each batch
    (which should use ``jax.device_put`` for async H2D transfer) and
    keeps ``size`` prepared batches in a deque. When the consumer pops
    a batch, the transfer has already completed or is finishing, so the
    train step never waits on H2D.
    """

    def __init__(
        self,
        source: Iterator[B],
        prepare_fn: Callable[[B], JB],
        *,
        cpu_prefetch: int = 4,
        device_prefetch: int = 2,
    ) -> None:
        self._cpu_iter: PrefetchIterator[B] | None = PrefetchIterator(
            source, cpu_prefetch
        )
        self._prepare = prepare_fn
        self._buf: deque[JB] = deque()
        self._device_prefetch = device_prefetch
        self._exhausted = False
        self._fill()

    def _fill(self) -> None:
        while len(self._buf) < self._device_prefetch and not self._exhausted:
            try:
                raw = next(self._cpu_iter)  # type: ignore[arg-type]
                self._buf.append(self._prepare(raw))
            except StopIteration:
                self._exhausted = True

    def __iter__(self) -> "DevicePrefetchIterator[B, JB]":
        return self

    def __next__(self) -> JB:
        if not self._buf:
            raise StopIteration
        item = self._buf.popleft()
        self._fill()
        return item

    def close(self) -> None:
        if self._cpu_iter is not None:
            self._cpu_iter.close()
            self._cpu_iter = None
        self._buf.clear()

    def __enter__(self) -> "DevicePrefetchIterator[B, JB]":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
