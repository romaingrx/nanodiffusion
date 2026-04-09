"""Named pretraining datasets.

Per-dataset modules self-register via :class:`Registry` side-effects on
import, so the top-level ``__init__`` only has to import them by name.
Adding a new dataset is one new file under this package plus one import
below.
"""

# Side-effect imports: each module runs a ``register_hf_parquet`` call at
# import time, populating ``DATASETS``. The tuple reference keeps the
# static analyzer from flagging the modules as unused while staying
# honest about why the import block exists.
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

# ``register`` is the legacy decorator name used by tests; alias to the
# registry's bound method for backward compatibility.
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
