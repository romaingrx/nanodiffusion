"""Smoke tests for ``load_runtime`` + ``warmup``."""

from pathlib import Path

import pytest

from nanodiffusion.inference import SampleConfigOverride, load_runtime, warmup


def test_load_runtime_uses_config_sample_with_no_override(
    saved_checkpoint: Path,
) -> None:
    runtime = load_runtime(saved_checkpoint)
    assert runtime.defaults.steps == 4
    assert runtime.defaults.max_length == 32


def test_load_runtime_override_replaces_named_fields(saved_checkpoint: Path) -> None:
    runtime = load_runtime(
        saved_checkpoint,
        overrides=SampleConfigOverride(steps=16, temperature=0.5),
    )
    assert runtime.defaults.steps == 16
    assert runtime.defaults.temperature == 0.5
    assert runtime.defaults.max_length == 32


def test_load_runtime_override_top_k_zero_is_respected(
    saved_checkpoint: Path,
) -> None:
    """Regression: ``top_k=0`` is a valid override; falsy guards must not
    treat it as "unset" and silently use the checkpoint value."""
    runtime = load_runtime(saved_checkpoint, overrides=SampleConfigOverride(top_k=0))
    assert runtime.defaults.top_k == 0


def test_load_runtime_missing_sidecar_errors(tmp_path: Path) -> None:
    empty = tmp_path / "no_sidecar"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):
        load_runtime(empty)


def test_warmup_default_covers_defaults_and_model_max_seq_len(
    saved_checkpoint: Path,
) -> None:
    """The default warmup primes both the configured ``max_length`` and the
    model's ``max_seq_len`` so the two most common request shapes are hot."""
    warmup(load_runtime(saved_checkpoint))


def test_warmup_explicit_lengths_overrides_default(saved_checkpoint: Path) -> None:
    warmup(load_runtime(saved_checkpoint), max_lengths=[16])
