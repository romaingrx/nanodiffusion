from click.testing import CliRunner

from nanodiffusion.cli import main
from nanodiffusion.cli.data import data_group
from nanodiffusion.cli.sample import sample_command


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
    assert "climbmix-400b" in result.output
    assert "fineweb-edu-10bt" in result.output


def test_data_download_unknown_dataset_errors() -> None:
    runner = CliRunner()
    result = runner.invoke(
        data_group,
        ["download", "--dataset", "definitely-missing", "--num-train", "1"],
    )
    assert result.exit_code != 0
    assert "definitely-missing" in (result.output + str(result.exception))
