"""Indexable chat-conversation sources.

The SFT data pipeline consumes whole :class:`Conversation` objects
(not raw text) and shuffles / oversamples them across an epoch. Sources
therefore expose a dataset-style ``len + getitem`` interface, in contrast
to :mod:`nanodiffusion.data.source`'s streaming ``TextSource`` protocol
used by pretrain.

Mirrors nanochat's ``tasks/common.py::Task`` abstraction, adapted to our
``Conversation`` schema and without coupling to the ``datasets`` library.
"""

import random
from typing import Protocol, runtime_checkable

from nanodiffusion.chat import Conversation


@runtime_checkable
class ChatSource(Protocol):
    """Finite, indexable stream of :class:`Conversation` objects.

    Implementations are expected to hold or lazily cache their contents
    in memory so repeated ``__getitem__`` access is cheap; the SFT loader
    assumes random access is O(1) to support per-epoch shuffles.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Conversation: ...


class InMemoryChatSource:
    """Test double: serves a fixed list of conversations."""

    def __init__(self, conversations: list[Conversation]) -> None:
        if not conversations:
            err = "InMemoryChatSource requires at least one conversation"
            raise ValueError(err)
        self._conversations = list(conversations)

    def __len__(self) -> int:
        return len(self._conversations)

    def __getitem__(self, index: int) -> Conversation:
        return self._conversations[index]


class TaskMixture:
    """Deterministically shuffled concatenation of multiple chat sources.

    Each entry in ``sources`` contributes all its conversations to the
    global index space; the final order is a seeded shuffle of all
    ``(source_idx, local_idx)`` pairs. Oversampling is done by passing
    the same source in multiple times, exactly like nanochat's
    ``TaskMixture``.

    The shuffle is built once in ``__init__`` and reused for the
    lifetime of the mixture. This gives stable, reproducible training
    order across runs with the same ``seed``, and fast index lookups
    without per-access randomness.
    """

    def __init__(self, sources: list[ChatSource], *, seed: int = 42) -> None:
        if not sources:
            err = "TaskMixture requires at least one source"
            raise ValueError(err)
        for i, s in enumerate(sources):
            if len(s) == 0:
                err = f"TaskMixture source {i} is empty"
                raise ValueError(err)
        self._sources = list(sources)
        index_map: list[tuple[int, int]] = [
            (src_idx, local_idx)
            for src_idx, source in enumerate(self._sources)
            for local_idx in range(len(source))
        ]
        rng = random.Random(seed)  # noqa: S311  # deterministic, not crypto
        rng.shuffle(index_map)
        self._index_map = index_map

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, index: int) -> Conversation:
        src_idx, local_idx = self._index_map[index]
        return self._sources[src_idx][local_idx]
