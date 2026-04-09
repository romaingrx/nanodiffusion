"""Project-wide filename and path constants.

Only strings that cross module boundaries live here. Single-site
constants (URLs, regex patterns, magic numbers) stay next to the code
that uses them.
"""

from pathlib import Path

MODEL_FILENAME = "model.eqx"
EMA_FILENAME = "ema.eqx"
OPT_STATE_FILENAME = "opt_state.eqx"
META_FILENAME = "meta.json"
CONFIG_SIDECAR_FILENAME = "config.yaml"
LATEST_LINK_NAME = "latest"
SCHEMA_PATH = Path("configs/config.schema.json")
