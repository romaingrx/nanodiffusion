from typing import TypeAlias

from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

__all__ = [
    "LogitBatch",
    "Logits",
    "Mask",
    "MaskBatch",
    "PRNGKeyArray",
    "Scalar",
    "TokenBatch",
    "Tokens",
]

Scalar: TypeAlias = Float[Array, ""]

# Token sequences: integer token IDs
Tokens: TypeAlias = Int[Array, " seq"]
TokenBatch: TypeAlias = Int[Array, "batch seq"]

# Model output logits
Logits: TypeAlias = Float[Array, "seq vocab"]
LogitBatch: TypeAlias = Float[Array, "batch seq vocab"]

# Boolean masks (True = masked / to predict)
Mask: TypeAlias = Bool[Array, " seq"]
MaskBatch: TypeAlias = Bool[Array, "batch seq"]
