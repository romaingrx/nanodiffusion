import pytest
from pydantic import ValidationError

from nanodiffusion.config import OptimizerHyperparams, SFTConfig, SFTDatasetConfig

_DATASETS = [SFTDatasetConfig(name="smoltalk")]


def test_sft_config_requires_explicit_datasets() -> None:
    """Removing the default_factory forces callers to name their mixture.

    A silent default drowned SFT runs in the wrong data — the review for
    PR #15 made this the explicit signal that no SFT run should start
    without saying which datasets.
    """
    with pytest.raises(ValidationError, match="datasets"):
        SFTConfig()  # pyright: ignore[reportCallIssue]


def test_sft_config_satisfies_optimizer_hyperparams_protocol() -> None:
    """SFTConfig is structurally compatible with make_optimizer's Protocol."""
    assert isinstance(SFTConfig(datasets=_DATASETS), OptimizerHyperparams)


def test_sft_config_rejects_empty_dataset_list() -> None:
    with pytest.raises(ValidationError, match="at least one entry"):
        SFTConfig(datasets=[])


def test_sft_config_rejects_max_steps_below_warmup() -> None:
    with pytest.raises(ValidationError, match="max_steps"):
        SFTConfig(datasets=_DATASETS, warmup_steps=100, max_steps=50)


def test_sft_dataset_config_rejects_non_positive_epochs() -> None:
    with pytest.raises(ValidationError, match="epochs"):
        SFTDatasetConfig(name="smoltalk", epochs=0)


def test_sft_config_rejects_non_positive_save_every() -> None:
    with pytest.raises(ValidationError, match="save_every"):
        SFTConfig(datasets=_DATASETS, save_every=0)


def test_sft_config_rejects_ema_decay_out_of_range() -> None:
    with pytest.raises(ValidationError, match="ema_decay"):
        SFTConfig(datasets=_DATASETS, ema_decay=1.5)
    with pytest.raises(ValidationError, match="ema_decay"):
        SFTConfig(datasets=_DATASETS, ema_decay=-0.01)
