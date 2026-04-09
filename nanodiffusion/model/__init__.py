"""Diffusion model types and construction helpers."""

from typing import cast

import equinox as eqx
import jax

from nanodiffusion.config import ModelConfig
from nanodiffusion.model._base import DiffusionModel
from nanodiffusion.model.transformer import Transformer


def transformer_skeleton(config: ModelConfig) -> Transformer:
    """Zero-cost :class:`Transformer` shape tree via ``filter_eval_shape``.

    ``eqx.filter_eval_shape`` traces the constructor abstractly: no
    parameter tensors are allocated, no PRNG normal draws happen, and
    the returned tree has ``ShapeDtypeStruct`` leaves that
    :func:`eqx.tree_deserialise_leaves` fills in from disk. Saves
    several seconds (and a full copy of the model's memory) at startup
    of every resume-from-checkpoint path. ``key`` is required by the
    constructor but never drawn from under ``filter_eval_shape``; any
    fixed seed works.
    """
    return cast(
        "Transformer",
        eqx.filter_eval_shape(Transformer, config, key=jax.random.PRNGKey(0)),
    )


__all__ = ["DiffusionModel", "Transformer", "transformer_skeleton"]
