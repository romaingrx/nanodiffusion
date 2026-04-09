"""Streaming pretraining data pipeline."""

from nanodiffusion.data.cursors import LoaderCursor, PretrainCursor, SFTCursor
from nanodiffusion.data.datasets import (
    DATASETS,
    DatasetFactory,
    DownloadOptions,
    get_dataset,
    parquet_from_huggingface,
    register,
)
from nanodiffusion.data.loader import (
    BatchOutput,
    JaxBatch,
    PrefetchIterator,
    prefetch,
    pretrain_loader,
)
from nanodiffusion.data.source import (
    InMemoryTextSource,
    ParquetTextSource,
    Split,
    TextSource,
)

__all__ = [
    "DATASETS",
    "BatchOutput",
    "DatasetFactory",
    "DownloadOptions",
    "InMemoryTextSource",
    "JaxBatch",
    "LoaderCursor",
    "ParquetTextSource",
    "PrefetchIterator",
    "PretrainCursor",
    "SFTCursor",
    "Split",
    "TextSource",
    "get_dataset",
    "parquet_from_huggingface",
    "prefetch",
    "pretrain_loader",
    "register",
]
