"""Streaming pretraining data loader.

Greedy concat with EOS document separators. Each batch carries a parallel
``segments`` array of per-row document ids that resets to 0 at the start of
every row, so a future intra-document attention masking pass can drop in
without changing the loader interface.

The loader produces ``numpy`` buffers; the training loop is responsible for
calling :meth:`BatchOutput.to_jax` (or :func:`jax.device_put` directly) right
before the jitted train step. JAX's host-to-device transfer is async on GPU
backends, so this gives free overlap with the previous step's compute. The
:class:`PrefetchIterator` adds an additional CPU-side overlap by running the
loader in a daemon thread behind a bounded queue.
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
    """One pretraining batch with arrays already on the JAX device.

    Same fields as :class:`BatchOutput` but with JAX arrays instead of
    numpy. Returned by :meth:`BatchOutput.to_jax` so the training loop has
    a single object to pass around (versus a tuple, which would drop the
    ``state`` field and break call sites whenever a new array field is
    added).
    """

    tokens: TokenBatch
    segments: SegmentBatch
    state: SourcePosition


@dataclass(frozen=True, slots=True)
class BatchOutput:
    """One pretraining batch.

    ``tokens`` is the ``x0`` for diffusion training. ``segments`` is the
    per-position document id within each row, starting at 0 and incrementing
    on every EOS boundary; it is unused by the current model but lets a
    follow-up issue add intra-document attention masking without changing
    this interface. ``state`` is the source position the most recently
    consumed batch came from, suitable for fast-forward resume.

    Shape, dtype, and segment-id invariants are validated in
    :meth:`__post_init__`. The jaxtyping import hook also catches shape
    mismatches at construction when running under tests, but the explicit
    ``__post_init__`` checks are the production safety net (the hook is
    not installed outside the test session).
    """

    tokens: NumpyTokenBatch
    segments: NumpySegmentBatch
    state: SourcePosition

    def __post_init__(self) -> None:
        if self.tokens.shape != self.segments.shape:
            msg = (
                "tokens and segments must have the same shape; got "
                f"{self.tokens.shape} vs {self.segments.shape}"
            )
            raise ValueError(msg)
        expected_ndim = 2
        if self.tokens.ndim != expected_ndim:
            msg = f"tokens must be 2D (batch, seq); got shape {self.tokens.shape}"
            raise ValueError(msg)
        if self.tokens.dtype != np.int32:
            msg = f"tokens must be int32; got {self.tokens.dtype}"
            raise ValueError(msg)
        if self.segments.dtype != np.int32:
            msg = f"segments must be int32; got {self.segments.dtype}"
            raise ValueError(msg)

    def to_jax(self) -> JaxBatch:
        """Move tokens and segments to the default JAX device.

        Call this in the main thread immediately before the JIT'd train
        step; ``jnp.asarray`` is async on GPU backends and overlaps the
        host-to-device transfer with the tail of the previous compute
        step.
        """
        return JaxBatch(
            tokens=jnp.asarray(self.tokens),
            segments=jnp.asarray(self.segments),
            state=self.state,
        )


def pretrain_loader(
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

    Algorithm:
      1. Pull a document batch from the source.
      2. Tokenize via :meth:`TokenizerLike.encode_batch` (multi-threaded,
         GIL-free for tiktoken).
      3. Append each doc + EOS to a rolling token buffer; assign each
         token a monotonically increasing per-doc segment id.
      4. When the buffer holds at least ``batch_size * seq_len`` tokens,
         drain that prefix into a numpy chunk, reshape to
         ``(batch_size, seq_len)``, and renumber the segment ids per row
         so the first segment of every row is 0.
      5. Yield :class:`BatchOutput`. The leftover tail carries forward.

    No padding, no cropping, 100% utilization. ``start`` / ``step`` shard
    the source at its native granularity for future data-parallel use.

    ``max_empty_passes`` guards against the silent infinite loop that would
    occur if the tokenizer returned an empty list for every input doc
    (e.g. wrong special-token config). The loader raises after that many
    consecutive source batches produce no tokens.
    """
    if batch_size <= 0 or seq_len <= 0:
        msg = f"batch_size and seq_len must be positive, got {batch_size}, {seq_len}"
        raise ValueError(msg)

    eos = tokenizer.eos_token_id
    chunk_size = batch_size * seq_len
    tokens_buf: deque[int] = deque()
    segments_buf: deque[int] = deque()
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
        while len(tokens_buf) < chunk_size:
            try:
                doc_batch, position = next(docs_iter)
            except StopIteration:
                # PEP 479: a bare StopIteration would become RuntimeError
                # inside this generator. Surface the source exhaustion
                # explicitly so resume code can react.
                return
            last_state = position
            encoded = tokenizer.encode_batch(doc_batch, num_threads=tokenizer_threads)
            produced = False
            for doc_tokens in encoded:
                if not doc_tokens:
                    continue
                produced = True
                tokens_buf.extend(doc_tokens)
                segments_buf.extend([next_doc_id] * len(doc_tokens))
                tokens_buf.append(eos)
                segments_buf.append(next_doc_id)
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

        # Drain chunk_size tokens from each deque in a single pass each.
        # popleft is O(1), so building the chunk is O(chunk_size) total
        # instead of O(remaining + chunk_size) like del list[:k].
        tokens_arr = np.fromiter(
            (tokens_buf.popleft() for _ in range(chunk_size)),
            dtype=np.int32,
            count=chunk_size,
        )
        segments_arr = np.fromiter(
            (segments_buf.popleft() for _ in range(chunk_size)),
            dtype=np.int32,
            count=chunk_size,
        )

        # Recycle next_doc_id when both buffers fully drain so the counter
        # stays small enough to fit in int32 during long runs.
        if not tokens_buf:
            next_doc_id = 0

        tokens_chunk = tokens_arr.reshape(batch_size, seq_len)
        segments_chunk = segments_arr.reshape(batch_size, seq_len)
        # Subtract each row's minimum so segments[i, 0] == 0. Using min
        # (rather than the first column) is robust to any future change
        # that ever permutes the buffer; today the buffer is monotonic so
        # min equals the first column anyway.
        segments_chunk = segments_chunk - segments_chunk.min(axis=1, keepdims=True)

        yield BatchOutput(
            tokens=tokens_chunk,
            segments=segments_chunk,
            state=last_state,
        )


class PrefetchIterator[T]:
    """Background producer with a bounded look-ahead window.

    Wraps any iterator and runs ``next(source)`` on a single worker thread
    via :class:`~concurrent.futures.ThreadPoolExecutor`. Up to ``size``
    items can be in flight at a time, so the consumer is never blocked on
    CPU work while the GPU is busy.

    The implementation deliberately stays inside the standard library:
    the executor handles thread lifecycle, exception propagation, and
    idempotent shutdown for us. Compared to a hand-rolled
    ``threading.Thread`` + ``queue.Queue`` loop, this removes the entire
    class of "did I drain the sentinel correctly", "did I leak the
    thread", and "is close() reentrant" bugs.

    Exceptions raised by the source surface from :meth:`__next__` via
    ``Future.result()``. ``StopIteration`` is preserved (the executor
    captures any ``BaseException``), so finite sources iterate cleanly.
    Calling :meth:`close` (or exiting the ``with`` block) shuts the
    executor down with ``cancel_futures=True``, waits for the in-flight
    item to finish, and clears the buffer; subsequent ``next()`` calls
    raise :class:`StopIteration`.

    .. note::
        This wrapper may turn out to be unnecessary. ``jax.device_put``
        on a GPU backend is already async, so the host-to-device copy of
        batch N+1 naturally overlaps with the train step on batch N as
        long as ``next(loader)`` is fast enough relative to the step.
        For ROM-15 we keep the explicit prefetch because it's cheap to
        carry, but if profiling in ROM-16 shows the loader is not the
        bottleneck this whole class can be deleted in favor of bare
        ``jax.device_put`` calls in the training loop.
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
        self._exhausted = False
        self._fill()

    def _fill(self) -> None:
        """Submit work until the look-ahead window is full or the source is dry."""
        while not self._exhausted and len(self._inflight) < self._size:
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
            # The source is finite and we just hit the end. Tear down the
            # remaining in-flight futures (which would also raise) so the
            # next call sees an empty queue and stops cleanly.
            self._exhausted = True
            self._inflight.clear()
            raise
        self._fill()
        return value

    def close(self) -> None:
        """Shut down the worker. Idempotent and safe to call from any thread."""
        # cancel_futures=True drops anything not yet picked up by the worker;
        # wait=True blocks until the in-flight item finishes. Together these
        # give us the documented atomic-shutdown semantics, no manual locks
        # or sentinel choreography required.
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._inflight.clear()
        self._exhausted = True

    def __enter__(self) -> "PrefetchIterator[T]":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def prefetch[T](it: Iterator[T], size: int = 4) -> PrefetchIterator[T]:
    """Wrap ``it`` in a :class:`PrefetchIterator` with the given look-ahead."""
    return PrefetchIterator(it, size)
