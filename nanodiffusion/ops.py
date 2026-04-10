"""Backend-dispatched tensor operators.

JAX's default lowerings are not uniformly best on every backend. This
module wraps the ops we care about with a small runtime dispatch so
model code stays hardware-agnostic while production runs still get
vendor-specific kernels. The public surface is a handful of pure
functions with stable shapes; each picks the fastest implementation it
can find at call time via ``jax.default_backend()``.

Adding a new backend-specialised op should stay in this file so
``nanodiffusion/model/*`` never has to know which platform it's
running on.
"""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        flash_attention as _tpu_flash_attention,
    )
except ImportError:
    _tpu_flash_attention = None


def attention(
    q: Float[Array, "heads seq head_dim"],
    k: Float[Array, "heads seq head_dim"],
    v: Float[Array, "heads seq head_dim"],
) -> Float[Array, "heads seq head_dim"]:
    """Dense multi-head self-attention with O(seq) memory on TPU.

    Inputs and output share the ``(num_heads, seq, head_dim)`` layout
    so callers can pivot their per-head state (RMSNorm, RoPE) once and
    hand it straight to this function. The scale is fixed at
    ``1/sqrt(head_dim)`` to match the conventional SDPA definition;
    callers that need a different scale can pre-scale ``q`` before
    calling.

    Backend dispatch:

    * **TPU**: :func:`jax.experimental.pallas.ops.tpu.flash_attention`.
      A blocked-softmax kernel that never materialises the
      ``[num_heads, seq, seq]`` score matrix, so attention memory is
      O(seq) instead of O(seq^2) and the step is also faster than the
      manual matmul+softmax+matmul triplet.
    * **CPU / GPU**: :func:`jax.nn.dot_product_attention`. Portable
      fallback; on GPU it further lowers to cuDNN Flash-SDPA when
      available, on CPU it is a plain einsum. On TPU this path is
      strictly worse than Pallas because XLA keeps the score matrix
      alive, so we never take it there.
    """
    _, _, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    if _tpu_flash_attention is not None and jax.default_backend() == "tpu":
        return _tpu_flash_attention(
            q[jnp.newaxis],
            k[jnp.newaxis],
            v[jnp.newaxis],
            sm_scale=scale,
        )[0]

    q_tnh = jnp.transpose(q, (1, 0, 2))
    k_tnh = jnp.transpose(k, (1, 0, 2))
    v_tnh = jnp.transpose(v, (1, 0, 2))
    out_tnh = jax.nn.dot_product_attention(q_tnh, k_tnh, v_tnh, scale=scale)
    return jnp.transpose(out_tnh, (1, 0, 2))
