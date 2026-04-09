from typing import TypeAlias

import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

__all__ = [
    "LogitBatch",
    "Logits",
    "Mask",
    "MaskBatch",
    "NumpyLossMaskBatch",
    "NumpySegmentBatch",
    "NumpyTokenBatch",
    "NumpyTokens",
    "PRNGKeyArray",
    "Prompt",
    "Scalar",
    "SegmentBatch",
    "TokenBatch",
    "Tokens",
]

Scalar: TypeAlias = Float[Array, ""]

# Token sequences: integer token IDs (JAX, on-device)
Tokens: TypeAlias = Int[Array, " seq"]
TokenBatch: TypeAlias = Int[Array, "batch seq"]
Prompt: TypeAlias = Int[Array, " prompt"]

# Per-position document segment id within a row, used for intra-document
# attention masking. JAX, on-device.
SegmentBatch: TypeAlias = Int[Array, "batch seq"]

# Model output logits
Logits: TypeAlias = Float[Array, "seq vocab"]
LogitBatch: TypeAlias = Float[Array, "batch seq vocab"]

# Boolean masks (True = masked / to predict)
Mask: TypeAlias = Bool[Array, " seq"]
MaskBatch: TypeAlias = Bool[Array, "batch seq"]

# Host-side numpy variants. The data loader builds these on the CPU; the
# training loop calls jnp.asarray (or jax.device_put) on them right before
# the JIT'd train step so HtoD overlaps with compute.
NumpyTokens: TypeAlias = Int[np.ndarray, " seq"]
NumpyTokenBatch: TypeAlias = Int[np.ndarray, "batch seq"]
NumpySegmentBatch: TypeAlias = Int[np.ndarray, "batch seq"]
NumpyLossMaskBatch: TypeAlias = Bool[np.ndarray, "batch seq"]
