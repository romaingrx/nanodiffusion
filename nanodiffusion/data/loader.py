"""Streaming pretraining data loader.

Greedy concat with EOS document separators. Each batch carries a parallel
``segments`` array of per-row document ids that resets to 0 at the start of
every row, so a future intra-document attention masking pass can drop in
without changing the loader interface.

The loader produces ``numpy`` buffers; the training loop is responsible for
calling :func:`jax.device_put` (or ``jnp.asarray``) right before the jitted
train step. JAX's host-to-device transfer is async on GPU backends, so this
gives free overlap with the previous step's compute. The :class:`PrefetchIterator`
adds an additional CPU-side overlap by running the loader in a daemon thread
behind a bounded queue.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

import numpy as np
from jaxtyping import Int

from nanodiffusion.data.source import SourcePosition, Split, TextSource
from nanodiffusion.tokenizer import Tokenizer

type NumpyTokenBatch = Int[np.ndarray, "batch seq"]
type NumpySegmentBatch = Int[np.ndarray, "batch seq"]


@dataclass(frozen=True, slots=True)
class BatchOutput:
    """One pretraining batch.

    ``tokens`` is the ``x0`` for diffusion training. ``segments`` is the
    per-position document id within each row, starting at 0 and incrementing
    on every EOS boundary; it is unused by the current model but lets a
    follow-up issue add intra-document attention masking without changing
    this interface. ``state`` is the source position the most recently
    consumed batch came from, suitable for fast-forward resume.
    """

    tokens: NumpyTokenBatch
    segments: NumpySegmentBatch
    state: SourcePosition


def pretrain_loader(
    source: TextSource,
    tokenizer: Tokenizer,
    *,
    batch_size: int,
    seq_len: int,
    split: Split,
    tokenizer_batch_size: int = 128,
    tokenizer_threads: int = 4,
    start: int = 0,
    step: int = 1,
    resume_state: SourcePosition | None = None,
) -> Iterator[BatchOutput]:
    """Greedy-concat pretrain loader with EOS document separators.

    Algorithm:
      1. Pull a document batch from the source.
      2. Tokenize via :meth:`Tokenizer.encode_batch` (multi-threaded, GIL-free).
      3. Append each doc + EOS to a rolling token buffer; assign each
         token a monotonically increasing per-doc segment id.
      4. When the buffer holds at least ``batch_size * seq_len`` tokens,
         slice that prefix, reshape to ``(batch_size, seq_len)``, and
         renumber the segment ids per row so the first segment of every
         row is 0.
      5. Yield :class:`BatchOutput`. The leftover tail carries forward.

    No padding, no cropping, 100% utilization. ``start`` / ``step`` shard
    the source at the row-group level for future data-parallel use.
    """
    if batch_size <= 0 or seq_len <= 0:
        msg = f"batch_size and seq_len must be positive, got {batch_size}, {seq_len}"
        raise ValueError(msg)

    eos = tokenizer.eos_token_id
    chunk_size = batch_size * seq_len
    tokens_buf: list[int] = []
    segments_buf: list[int] = []
    next_doc_id = 0
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
            doc_batch, position = next(docs_iter)
            last_state = position
            encoded = tokenizer.encode_batch(doc_batch, num_threads=tokenizer_threads)
            for doc_tokens in encoded:
                if not doc_tokens:
                    continue
                tokens_buf.extend(doc_tokens)
                segments_buf.extend([next_doc_id] * len(doc_tokens))
                tokens_buf.append(eos)
                segments_buf.append(next_doc_id)
                next_doc_id += 1

        tokens_arr = np.asarray(tokens_buf[:chunk_size], dtype=np.int32)
        segments_arr = np.asarray(segments_buf[:chunk_size], dtype=np.int32)
        del tokens_buf[:chunk_size]
        del segments_buf[:chunk_size]

        tokens_chunk = tokens_arr.reshape(batch_size, seq_len)
        segments_chunk = segments_arr.reshape(batch_size, seq_len)
        # Subtract each row's first segment id so segments[i, 0] == 0.
        # The original ids are monotonic within a row, so this preserves
        # the increment-at-EOS structure while normalizing the offset.
        segments_chunk = segments_chunk - segments_chunk[:, :1]

        yield BatchOutput(
            tokens=tokens_chunk,
            segments=segments_chunk,
            state=last_state,
        )


_SENTINEL: Any = object()


class PrefetchIterator[T]:
    """Daemon-thread producer with a bounded queue.

    Wraps any iterator. The worker thread tokenizes, packs, and enqueues
    the next ``size`` batches while the main thread is busy training.
    Exceptions raised in the worker propagate to ``__next__`` on the
    consumer side. Calling :meth:`close` (or exiting the ``with`` block)
    sets a stop flag, drains the queue, and joins the thread.
    """

    def __init__(self, source_iter: Iterator[T], size: int) -> None:
        if size <= 0:
            msg = f"prefetch size must be positive, got {size}"
            raise ValueError(msg)
        self._source = source_iter
        self._queue: Queue[Any] = Queue(maxsize=size)
        self._stop = Event()
        self._closed = False
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            for item in self._source:
                if self._stop.is_set():
                    return
                self._queue.put(item)
        except Exception as exc:  # noqa: BLE001
            self._queue.put(exc)
        finally:
            self._queue.put(_SENTINEL)

    def __iter__(self) -> "PrefetchIterator[T]":
        return self

    def __next__(self) -> T:
        item = self._queue.get()
        if item is _SENTINEL:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item  # pyright: ignore[reportReturnType]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        # Drain so a worker blocked on put() can exit.
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        self._thread.join(timeout=2.0)

    def __enter__(self) -> "PrefetchIterator[T]":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def prefetch[T](it: Iterator[T], size: int = 4) -> PrefetchIterator[T]:
    """Wrap ``it`` in a :class:`PrefetchIterator` with the given queue depth."""
    return PrefetchIterator(it, size)
