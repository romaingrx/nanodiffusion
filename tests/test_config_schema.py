"""Schema drift + debug-config validation.

``configs/config.schema.json`` is derived from
``Config.model_json_schema()``; these tests make sure the checked-in
file stays in sync and that every YAML under ``configs/`` actually
validates against the live pydantic model.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nanodiffusion.cli.config import schema_document
from nanodiffusion.config import Config
from nanodiffusion.constants import SCHEMA_PATH

_REPO_ROOT = Path(__file__).parent.parent
_SCHEMA_PATH = _REPO_ROOT / SCHEMA_PATH
_CONFIGS_DIR = _REPO_ROOT / "configs"


def test_checked_in_schema_matches_current_config_class() -> None:
    """Fails when ``config.py`` drifted from the committed schema file.

    Regenerate via ``uv run nanodiffusion config gen-schema``.
    """
    expected = json.dumps(schema_document(), indent=2) + "\n"
    on_disk = _SCHEMA_PATH.read_text()
    assert on_disk == expected, (
        f"{_SCHEMA_PATH} is stale; regenerate with "
        "`uv run nanodiffusion config gen-schema`."
    )


def test_schema_and_configs_do_not_expose_dropout_rate() -> None:
    schema = json.loads(_SCHEMA_PATH.read_text())
    model_schema = schema["$defs"]["ModelConfig"]
    assert "dropout_rate" not in model_schema["properties"]
    for yaml_path in sorted(_CONFIGS_DIR.glob("*.yaml")):
        assert "dropout_rate" not in yaml_path.read_text()


@pytest.mark.parametrize(
    "yaml_path",
    sorted(p for p in _CONFIGS_DIR.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_configs_directory_yaml_validates(yaml_path: Path) -> None:
    """Every YAML in ``configs/`` must load cleanly into :class:`Config`."""
    try:
        Config.from_yaml(yaml_path)
    except ValidationError as exc:
        pytest.fail(f"{yaml_path} failed to validate: {exc}")
