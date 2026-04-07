from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from nanodiffusion.cli import main
from nanodiffusion.cli.data import data_group
from nanodiffusion.cli.sample import sample_command
from nanodiffusion.data.datasets import DATASETS, DownloadOptions


def test_main_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "sample" in result.output
    assert "data" in result.output


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
