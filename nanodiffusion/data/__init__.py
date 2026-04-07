"""Streaming pretraining data pipeline."""

from nanodiffusion.data.datasets import (
    DATASETS,
    Dataset,
    get,
    parquet_from_huggingface,
    register,
)
from nanodiffusion.data.loader import (
    BatchOutput,
    NumpySegmentBatch,
    NumpyTokenBatch,
    PrefetchIterator,
    prefetch,
    pretrain_loader,
)
from nanodiffusion.data.source import (
    InMemoryTextSource,
    ParquetTextSource,
    SourcePosition,
    Split,
    TextSource,
)

__all__ = [
    "DATASETS",
    "BatchOutput",
    "Dataset",
    "InMemoryTextSource",
    "NumpySegmentBatch",
    "NumpyTokenBatch",
    "ParquetTextSource",
    "PrefetchIterator",
    "SourcePosition",
    "Split",
    "TextSource",
    "get",
    "parquet_from_huggingface",
    "prefetch",
    "pretrain_loader",
    "register",
]
