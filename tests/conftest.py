from pathlib import Path

import pytest
import yaml


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    data = {
        "model": {
            "num_layers": 4,
            "hidden_dim": 256,
            "num_heads": 4,
        },
        "train": {
            "batch_size": 8,
            "max_steps": 100,
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return p
