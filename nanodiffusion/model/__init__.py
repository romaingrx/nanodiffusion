"""Diffusion model types and construction helpers."""

from typing import cast

import equinox as eqx
import jax

from nanodiffusion.config import ModelConfig
from nanodiffusion.model._base import DiffusionModel
from nanodiffusion.model.transformer import Transformer


def transformer_skeleton(config: ModelConfig) -> Transformer:
    """Zero-cost :class:`Transformer` shape tree for deserialisation.

    ``eqx.filter_eval_shape`` traces the constructor abstractly — no
    parameter tensors are allocated and no PRNG draws happen — so the
    returned tree has ``ShapeDtypeStruct`` leaves that
    :func:`eqx.tree_deserialise_leaves` can fill in from disk. ``key``
    is required by the constructor signature but is not actually drawn.
    """
    return cast(
        "Transformer",
        eqx.filter_eval_shape(Transformer, config, key=jax.random.PRNGKey(0)),
    )


__all__ = ["DiffusionModel", "Transformer", "transformer_skeleton"]
