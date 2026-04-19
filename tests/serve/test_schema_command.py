"""Schema CLI round-trip + checked-in file drift test."""

import json
from pathlib import Path

from click.testing import CliRunner

from nanodiffusion.cli.serve import schema_command
from nanodiffusion.serve.protocol import EXPORTED_MODELS, schema_document


def test_schema_command_writes_all_models(tmp_path: Path) -> None:
    output = tmp_path / "protocol.json"
    result = CliRunner().invoke(schema_command, ["--output", str(output)])
    assert result.exit_code == 0, result.output

    doc = json.loads(output.read_text())
    defs = doc["$defs"]
    for model in EXPORTED_MODELS:
        assert model.__name__ in defs


def test_checked_in_schema_matches_current_protocol() -> None:
    """The committed ``schemas/protocol.json`` must stay in lockstep with the
    live Pydantic definitions. Regenerate with ``nanodiffusion schema``."""
    on_disk = Path("schemas/protocol.json").read_text()
    fresh = json.dumps(schema_document(), indent=2) + "\n"
    assert on_disk == fresh, (
        "schemas/protocol.json is stale — run `nanodiffusion schema` to regenerate"
    )
