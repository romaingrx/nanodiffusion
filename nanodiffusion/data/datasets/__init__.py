"""Named pretraining datasets.

Adding a new dataset is one new file under this package plus one
side-effect import below.
"""

# Side-effect imports: each module registers itself at import time.
# The tuple below anchors the imports against an unused-import warning.
from nanodiffusion.data.datasets import climbmix_400b, fineweb_edu
from nanodiffusion.data.datasets._base import (
    DATASETS,
    DatasetFactory,
    DownloadOptions,
    download_shards,
    download_with_backoff,
    get_dataset,
    parquet_from_huggingface,
    register_hf_parquet,
    validate_parquet,
)

_SIDE_EFFECT_MODULES = (climbmix_400b, fineweb_edu)

register = DATASETS.register


__all__ = [
    "DATASETS",
    "DatasetFactory",
    "DownloadOptions",
    "download_shards",
    "download_with_backoff",
    "get_dataset",
    "parquet_from_huggingface",
    "register",
    "register_hf_parquet",
    "validate_parquet",
]
