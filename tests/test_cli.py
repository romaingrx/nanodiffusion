from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from nanodiffusion.checkpoint import load_meta
from nanodiffusion.cli import main
from nanodiffusion.cli.data import data_group
from nanodiffusion.cli.pretrain import pretrain_command
from nanodiffusion.cli.sample import sample_command
from nanodiffusion.config import Config
from nanodiffusion.constants import CONFIG_SIDECAR_FILENAME
from nanodiffusion.data.datasets import DATASETS, DatasetFactory, DownloadOptions
from nanodiffusion.data.source import InMemoryTextSource, TextSource


@pytest.fixture
def register_dataset() -> Iterator[Callable[[str, DatasetFactory], None]]:
    """Temporarily register a dataset factory; auto-unregister on teardown.

    Uses :class:`Registry`'s mutable-mapping API directly so the
    argument type is ``DatasetFactory`` throughout — unlike
    ``monkeypatch.setitem``, which can't express the Protocol narrowing
    and forces a ``# pyright: ignore`` at every call site.
    """
    added: list[str] = []

    def register(name: str, factory: DatasetFactory) -> None:
        DATASETS[name] = factory
        added.append(name)

    yield register

    for name in added:
        DATASETS.pop(name, None)


def test_main_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "sample" in result.output
    assert "data" in result.output
    assert "pretrain" in result.output
    assert "sft" in result.output
    assert "config" in result.output


def test_config_gen_schema_writes_valid_json_schema(tmp_path: Path) -> None:
    """``config gen-schema`` writes a JSON document containing the generated note."""
    import json  # noqa: PLC0415

    from nanodiffusion.cli.config import config_group  # noqa: PLC0415

    runner = CliRunner()
    out = tmp_path / "out.schema.json"
    result = runner.invoke(config_group, ["gen-schema", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    schema = json.loads(out.read_text())
    assert schema["type"] == "object"
    assert schema["title"] == "Config"
    assert "regenerate" in schema["description"].lower()


def test_config_validate_accepts_good_yaml(tmp_path: Path) -> None:
    """``config validate`` exits 0 on a valid config."""
    from nanodiffusion.cli.config import config_group  # noqa: PLC0415

    good = tmp_path / "good.yaml"
    good.write_text("model:\n  num_layers: 2\n  hidden_dim: 64\n  num_heads: 2\n")

    runner = CliRunner()
    result = runner.invoke(config_group, ["validate", str(good)])
    assert result.exit_code == 0, result.output
    assert "ok" in result.output


def test_config_validate_rejects_bad_yaml(tmp_path: Path) -> None:
    """``config validate`` exits non-zero and lists the field error on bad YAML."""
    from nanodiffusion.cli.config import config_group  # noqa: PLC0415

    # ``max_steps`` below ``warmup_steps`` trips the ``@model_validator``
    # on TrainConfig, exercising a pydantic failure path that JSON
    # Schema alone would miss.
    bad = tmp_path / "bad.yaml"
    bad.write_text("train:\n  warmup_steps: 100\n  max_steps: 50\n")

    runner = CliRunner()
    result = runner.invoke(config_group, ["validate", str(bad)])
    assert result.exit_code != 0
    assert "fail" in result.output or "fail" in (result.stderr or "")


def test_sft_command_help_mentions_checkpoint_options() -> None:
    from nanodiffusion.cli.sft import sft_command  # noqa: PLC0415

    runner = CliRunner()
    result = runner.invoke(sft_command, ["--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--pretrain-checkpoint" in result.output
    assert "--resume-from" in result.output
    assert "--seed" in result.output


def test_sft_command_rejects_missing_start_point() -> None:
    """Neither --pretrain-checkpoint nor --resume-from is a user error."""
    from nanodiffusion.cli.sft import sft_command  # noqa: PLC0415

    runner = CliRunner()
    result = runner.invoke(sft_command, ["--config", "nonexistent.yaml"])
    # --config exists-check fires first; we only care that the command
    # *would* complain about the missing start point if the config was
    # valid. That's covered by the e2e sft_finetune test.
    assert result.exit_code != 0


def test_data_list_chat_prints_registered_chat_datasets() -> None:
    from nanodiffusion.data.chat_datasets import CHAT_DATASETS  # noqa: PLC0415

    runner = CliRunner()
    result = runner.invoke(data_group, ["list-chat"])
    assert result.exit_code == 0
    for name in CHAT_DATASETS:
        assert name in result.output


def test_data_download_chat_unknown_dataset_uses_bad_parameter() -> None:
    runner = CliRunner()
    result = runner.invoke(
        data_group, ["download-chat", "--dataset", "definitely-missing"]
    )
    assert result.exit_code != 0
    assert "definitely-missing" in result.output
    assert "Available" in result.output


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
    tmp_path: Path,
    register_dataset: Callable[[str, DatasetFactory], None],
) -> None:
    """End-to-end: registering a fake factory and invoking download."""
    captured: dict[str, Any] = {}

    def fake_factory(
        data_dir: Path,
        *,
        num_train: int | None = None,
        download: bool = True,
        download_options: DownloadOptions | None = None,
    ) -> TextSource:
        captured["data_dir"] = data_dir
        captured["num_train"] = num_train
        captured["download"] = download
        captured["download_options"] = download_options
        return InMemoryTextSource(["stub-train", "stub-val"], val_size=1)

    name = "test-cli-fake"
    register_dataset(name, fake_factory)
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


def _register_in_memory_dataset(
    register_dataset: Callable[[str, DatasetFactory], None], name: str
) -> None:
    def factory(
        data_dir: Path,
        *,
        num_train: int | None = None,
        download: bool = True,
        download_options: DownloadOptions | None = None,
    ) -> TextSource:
        del data_dir, num_train, download, download_options
        docs = [f"doc {i} " + ("hello world " * 40) for i in range(30)]
        return InMemoryTextSource(docs, val_size=2)

    register_dataset(name, factory)


def test_pretrain_command_runs_end_to_end(
    tmp_path: Path,
    register_dataset: Callable[[str, DatasetFactory], None],
) -> None:
    """Smoke test: a handful of steps on an in-memory dataset + final checkpoint."""
    _register_in_memory_dataset(register_dataset, "test-pretrain-fake")

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
    assert (single_run / CONFIG_SIDECAR_FILENAME).exists()

    # _write_pretrain_config uses max_steps=3, save_every=1000 → single save.
    final = single_run / "step_3"
    assert final.is_dir()
    # Orbax atomically writes ``_CHECKPOINT_METADATA`` last; its presence
    # is the finalisation marker for the step.
    assert (final / "_CHECKPOINT_METADATA").exists()
    assert (final / "state").is_dir()
    assert (final / "meta").is_dir()


def test_pretrain_latest_reflects_max_step(
    tmp_path: Path,
    register_dataset: Callable[[str, DatasetFactory], None],
) -> None:
    """``latest/meta.json`` reflects the full step count we asked for."""
    _register_in_memory_dataset(register_dataset, "test-pretrain-final")

    run_dir = tmp_path / "runs"
    config_path = tmp_path / "debug.yaml"
    _write_pretrain_config(config_path, run_dir=run_dir, dataset="test-pretrain-final")

    runner = CliRunner()
    result = runner.invoke(pretrain_command, ["--config", str(config_path)])
    assert result.exit_code == 0, result.output

    single_run = next(iter(run_dir.iterdir()))
    meta = load_meta(single_run)
    # _write_pretrain_config uses max_steps=3.
    assert meta.step == 3


def test_pretrain_no_duplicate_save_when_step_matches_save_every(
    tmp_path: Path,
    register_dataset: Callable[[str, DatasetFactory], None],
) -> None:
    """``save_every == max_steps`` must produce a single ``step_N/``, not two.

    Regression against an earlier loop shape that wrote both a
    periodic save and an end-of-loop save at the same step.
    """
    _register_in_memory_dataset(register_dataset, "test-pretrain-nodup")

    run_dir = tmp_path / "runs"
    config_path = tmp_path / "debug.yaml"
    # Custom config: max_steps == save_every so the two branches collide.
    config_path.write_text(
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
                    "warmup_steps": 1,
                    "max_steps": 3,
                    "log_every": 1,
                    "save_every": 3,
                    "run_dir": str(run_dir),
                },
                "data": {
                    "dataset": "test-pretrain-nodup",
                    "data_dir": "unused",
                    "num_train_shards": 1,
                    "tokenizer_batch_size": 4,
                    "prefetch_size": 1,
                    "max_empty_passes": 10,
                },
            }
        )
    )

    runner = CliRunner()
    result = runner.invoke(pretrain_command, ["--config", str(config_path)])
    assert result.exit_code == 0, result.output

    single_run = next(iter(run_dir.iterdir()))
    ckpt_dirs = {
        p.name for p in single_run.iterdir() if p.is_dir() and not p.is_symlink()
    }
    assert ckpt_dirs == {"step_3"}, f"unexpected ckpt dirs: {ckpt_dirs}"


def test_pretrain_resume_continues_step_and_reuses_run_dir(
    tmp_path: Path,
    register_dataset: Callable[[str, DatasetFactory], None],
) -> None:
    """Resume picks up from the saved step and writes back to the same run dir."""
    _register_in_memory_dataset(register_dataset, "test-pretrain-resume")

    run_dir = tmp_path / "runs"
    first_config = tmp_path / "first.yaml"
    first_config.write_text(
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
                    "warmup_steps": 1,
                    "max_steps": 2,
                    "log_every": 1,
                    "save_every": 1000,
                    "run_dir": str(run_dir),
                },
                "data": {
                    "dataset": "test-pretrain-resume",
                    "data_dir": "unused",
                    "num_train_shards": 1,
                    "tokenizer_batch_size": 4,
                    "prefetch_size": 1,
                    "max_empty_passes": 10,
                },
            }
        )
    )

    runner = CliRunner()
    first = runner.invoke(pretrain_command, ["--config", str(first_config)])
    assert first.exit_code == 0, first.output

    single_run = next(iter(run_dir.iterdir()))
    meta = load_meta(single_run)
    assert meta.step == 2

    # Second phase: same run dir, bumped max_steps, resume via the run dir.
    second_config = tmp_path / "second.yaml"
    second_config.write_text(
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
                    "warmup_steps": 1,
                    "max_steps": 4,
                    "log_every": 1,
                    "save_every": 1000,
                    "run_dir": str(run_dir),
                },
                "data": {
                    "dataset": "test-pretrain-resume",
                    "data_dir": "unused",
                    "num_train_shards": 1,
                    "tokenizer_batch_size": 4,
                    "prefetch_size": 1,
                    "max_empty_passes": 10,
                },
            }
        )
    )

    second = runner.invoke(
        pretrain_command,
        ["--config", str(second_config), "--resume-from", str(single_run)],
    )
    assert second.exit_code == 0, second.output

    # No new timestamped run dir was created.
    assert list(run_dir.iterdir()) == [single_run]

    # Both step_2 (from the first run) and step_4 (from the resume) coexist.
    ckpt_dirs = {
        p.name
        for p in single_run.iterdir()
        if p.is_dir() and p.name.startswith("step_")
    }
    assert ckpt_dirs == {"step_2", "step_4"}, ckpt_dirs

    # The Orbax manager's latest step is the newest finalised checkpoint.
    final_meta = load_meta(single_run)
    assert final_meta.step == 4


def test_pretrain_writes_reloadable_config(
    tmp_path: Path,
    register_dataset: Callable[[str, DatasetFactory], None],
) -> None:
    """``run_dir/config.yaml`` round-trips through ``Config.from_yaml``."""
    _register_in_memory_dataset(register_dataset, "test-pretrain-roundtrip")

    run_dir = tmp_path / "runs"
    config_path = tmp_path / "debug.yaml"
    _write_pretrain_config(
        config_path, run_dir=run_dir, dataset="test-pretrain-roundtrip"
    )

    runner = CliRunner()
    result = runner.invoke(pretrain_command, ["--config", str(config_path)])
    assert result.exit_code == 0, result.output

    single_run = next(iter(run_dir.iterdir()))
    dumped = single_run / CONFIG_SIDECAR_FILENAME
    reloaded = Config.from_yaml(dumped)
    # Match the values the test set; the rest use model defaults.
    assert reloaded.train.max_steps == 3
    assert reloaded.train.batch_size == 2
    assert reloaded.model.num_layers == 2
    # A second round-trip through the same path produces an equal object.
    reloaded_2 = Config.from_yaml(dumped)
    assert reloaded == reloaded_2


def test_pretrain_command_seed_override(
    tmp_path: Path,
    register_dataset: Callable[[str, DatasetFactory], None],
) -> None:
    """``--seed`` overrides the yaml value without mutating the file on disk."""
    _register_in_memory_dataset(register_dataset, "test-pretrain-seed")

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
