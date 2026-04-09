"""Supervised fine-tuning with role-aware masking."""

from nanodiffusion.sft.loss import compute_sft_loss, sft_forward_mask
from nanodiffusion.sft.train import (
    SFTTrainStepFn,
    make_sft_train_step,
    sft_finetune,
)

__all__ = [
    "SFTTrainStepFn",
    "compute_sft_loss",
    "make_sft_train_step",
    "sft_finetune",
    "sft_forward_mask",
]
