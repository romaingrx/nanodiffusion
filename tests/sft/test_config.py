import pytest
from pydantic import ValidationError

from nanodiffusion.config import OptimizerHyperparams, SFTConfig, SFTDatasetConfig


def test_sft_config_default_dataset_mixture_matches_issue() -> None:
    """Defaults mirror nanochat's scoped-down mixture from ROM-17.

    SmolTalk once, GSM8K x4, identity x2 — a regression against any
    silent change to the oversampling ratios.
    """
    cfg = SFTConfig()
    names = {d.name: d.epochs for d in cfg.datasets}
    assert names == {"smoltalk": 1, "gsm8k": 4, "identity": 2}


def test_sft_config_satisfies_optimizer_hyperparams_protocol() -> None:
    """SFTConfig is structurally compatible with make_optimizer's Protocol."""
    assert isinstance(SFTConfig(), OptimizerHyperparams)


def test_sft_config_rejects_empty_dataset_list() -> None:
    with pytest.raises(ValidationError, match="at least one entry"):
        SFTConfig(datasets=[])


def test_sft_config_rejects_max_steps_below_warmup() -> None:
    with pytest.raises(ValidationError, match="max_steps"):
        SFTConfig(warmup_steps=100, max_steps=50)


def test_sft_dataset_config_rejects_non_positive_epochs() -> None:
    with pytest.raises(ValidationError, match="epochs"):
        SFTDatasetConfig(name="smoltalk", epochs=0)


def test_sft_config_rejects_non_positive_save_every() -> None:
    with pytest.raises(ValidationError, match="save_every"):
        SFTConfig(save_every=0)


def test_sft_config_rejects_ema_decay_out_of_range() -> None:
    with pytest.raises(ValidationError, match="ema_decay"):
        SFTConfig(ema_decay=1.5)
    with pytest.raises(ValidationError, match="ema_decay"):
        SFTConfig(ema_decay=-0.01)
