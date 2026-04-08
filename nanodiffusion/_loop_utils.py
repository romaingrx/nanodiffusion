"""Shared run-directory helpers for pretrain and SFT training loops.

Both drivers materialize a per-run directory under ``<run_dir>/<run_id>/``
with the resolved config file dropped in, then write periodic checkpoints
alongside a ``latest`` symlink. Extracting the tiny bits that don't vary
by paradigm into this module keeps :mod:`nanodiffusion.pretrain.train`
and :mod:`nanodiffusion.sft.train` focused on their respective loop
bodies without the two files drifting on filesystem conventions.
"""

import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel


def make_run_id() -> str:
    """UTC timestamp run id, e.g. ``20260408-193015``."""
    return datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")


def write_config(run_dir: Path, config: BaseModel) -> None:
    """Dump a resolved pydantic config to ``run_dir/config.yaml``.

    Uses ``model_dump(mode="json")`` so ``Path`` and other non-yaml
    types serialize cleanly — matches what :func:`yaml.safe_load` will
    accept back when we reload the config at sampling time.
    """
    (run_dir / "config.yaml").write_text(yaml.dump(config.model_dump(mode="json")))
