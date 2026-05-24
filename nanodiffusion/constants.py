"""Project-wide filename and path constants.

Only strings that cross module boundaries live here. Single-site
constants (URLs, regex patterns, magic numbers) stay next to the code
that uses them.
"""

from pathlib import Path

CONFIG_SIDECAR_FILENAME = "config.yaml"
SCHEMA_PATH = Path("configs/config.schema.json")
