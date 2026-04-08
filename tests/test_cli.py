from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from nanodiffusion.cli import main
from nanodiffusion.cli.data import data_group
from nanodiffusion.cli.pretrain import pretrain_command
from nanodiffusion.cli.sample import sample_command
from nanodiffusion.data.datasets import DATASETS, DownloadOptions
from nanodiffusion.data.source import InMemoryTextSource


def test_main_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "sample" in result.output
    assert "data" in result.output
    assert "pretrain" in result.output


def test_sample_command_help() -> None:
    runner = CliRunner()
    result = runner.invoke(sample_command, ["--help"])
    assert result.exit_code == 0
    assert "Generate text" in result.output


def test_data_group_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(data_group, ["--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "download" in result.output


def test_data_list_prints_registered_datasets() -> None:
    runner = CliRunner()
    result = runner.invoke(data_group, ["list"])
    assert result.exit_code == 0
    for name in DATASETS:
        assert name in result.output


def test_data_list_includes_factory_docstring() -> None:
    """List entries surface the first line of each factory's docstring."""
    runner = CliRunner()
    result = runner.invoke(data_group, ["list"])
    assert result.exit_code == 0
    # climbmix factory's docstring starts with "ClimbMix-400B"
    assert "ClimbMix-400B" in result.output


def test_data_download_unknown_dataset_uses_bad_parameter() -> None:
    """Unknown dataset must surface as a friendly BadParameter error."""
    runner = CliRunner()
    result = runner.invoke(
        data_group,
        ["download", "--dataset", "definitely-missing", "--num-train", "1"],
    )
    assert result.exit_code != 0
    # Click prints BadParameter as a friendly message, not a Python traceback
    assert "definitely-missing" in result.output
    assert "Available" in result.output


def test_data_download_invokes_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: registering a fake factory and invoking download."""
    captured: dict[str, Any] = {}

    def fake_factory(
        data_dir: Path,
        *,
        num_train: int | None = None,
        download: bool = True,
        download_options: DownloadOptions | None = None,
    ) -> object:
        captured["data_dir"] = data_dir
        captured["num_train"] = num_train
        captured["download"] = download
        captured["download_options"] = download_options
        return object()

    name = "test-cli-fake"
    monkeypatch.setitem(DATASETS, name, fake_factory)  # pyright: ignore[reportArgumentType]
    runner = CliRunner()
    result = runner.invoke(
        data_group,
        [
            "download",
            "--dataset",
            name,
            "--num-train",
            "7",
            "--data-dir",
            str(tmp_path / "out"),
            "--retries",
            "3",
            "--timeout",
            "30",
            "--num-workers",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["num_train"] == 7
    assert captured["download"] is True
    assert captured["data_dir"] == tmp_path / "out"
    options = captured["download_options"]
    assert isinstance(options, DownloadOptions)
    assert options.retries == 3
    assert options.timeout == 30.0
    assert options.num_workers == 2
    assert "Downloaded 7" in result.output


def test_pretrain_command_help_mentions_config_option() -> None:
    runner = CliRunner()
    result = runner.invoke(pretrain_command, ["--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--seed" in result.output
    assert "--resume-from" in result.output


def _write_pretrain_config(path: Path, *, run_dir: Path, dataset: str) -> None:
    path.write_text(
        yaml.dump(
            {
                "model": {
                    "vocab_size": 50264,
                    "num_layers": 2,
                    "hidden_dim": 64,
                    "num_heads": 4,
                    "max_seq_len": 32,
                },
                "train": {
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "warmup_steps": 2,
                    "max_steps": 3,
                    "log_every": 1,
                    "save_every": 1000,
                    "run_dir": str(run_dir),
                },
                "data": {
                    "dataset": dataset,
                    "data_dir": "unused",
                    "num_train_shards": 1,
                    "tokenizer_batch_size": 4,
                    "prefetch_size": 1,
                    "max_empty_passes": 10,
                },
            }
        )
    )


def _register_in_memory_dataset(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    def factory(
        data_dir: Path,
        *,
        num_train: int | None = None,
        download: bool = True,
        download_options: DownloadOptions | None = None,
    ) -> InMemoryTextSource:
        del data_dir, num_train, download, download_options
        docs = [f"doc {i} " + ("hello world " * 40) for i in range(30)]
        return InMemoryTextSource(docs, val_size=2)

    monkeypatch.setitem(DATASETS, name, factory)  # pyright: ignore[reportArgumentType]


def test_pretrain_command_runs_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke test: a handful of steps on an in-memory dataset + final checkpoint."""
    _register_in_memory_dataset(monkeypatch, "test-pretrain-fake")

    run_dir = tmp_path / "runs"
    config_path = tmp_path / "debug.yaml"
    _write_pretrain_config(config_path, run_dir=run_dir, dataset="test-pretrain-fake")

    runner = CliRunner()
    result = runner.invoke(
        pretrain_command, ["--config", str(config_path), "--seed", "0"]
    )
    assert result.exit_code == 0, result.output

    runs = list(run_dir.iterdir())
    assert len(runs) == 1
    single_run = runs[0]
    assert (single_run / "config.yaml").exists()

    checkpoints = sorted(p for p in single_run.iterdir() if p.is_dir())
    assert checkpoints, "expected at least a final checkpoint"
    final = checkpoints[-1]
    assert (final / "model.eqx").exists()
    assert (final / "ema.eqx").exists()
    assert (final / "opt_state.eqx").exists()
    assert (final / "meta.json").exists()
    assert (final / "config.yaml").exists()


def test_pretrain_command_seed_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--seed`` overrides the yaml value without mutating the file on disk."""
    _register_in_memory_dataset(monkeypatch, "test-pretrain-seed")

    config_path = tmp_path / "debug.yaml"
    _write_pretrain_config(
        config_path, run_dir=tmp_path / "runs", dataset="test-pretrain-seed"
    )
    original = config_path.read_text()

    runner = CliRunner()
    result = runner.invoke(
        pretrain_command, ["--config", str(config_path), "--seed", "999"]
    )
    assert result.exit_code == 0, result.output
    # Config file on disk is not rewritten.
    assert config_path.read_text() == original
