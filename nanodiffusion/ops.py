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

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        flash_attention as _tpu_flash_attention,
    )
except ImportError:
    _tpu_flash_attention = None


def cast_dtype[M: eqx.Module](model: M, dtype: type) -> M:
    """Cast every inexact (float) leaf of ``model`` to ``dtype``.

    Integer leaves, static fields, and non-array leaves are left
    untouched. Intended for mixed-precision training: cast the
    model to bf16 before the forward pass so matmuls run at
    native TPU speed, while the caller keeps fp32 master weights
    for stable optimizer updates.
    """
    return jax.tree.map(
        lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x,
        model,
        is_leaf=eqx.is_inexact_array,
    )


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

    * **Single-device TPU**:
      :func:`jax.experimental.pallas.ops.tpu.flash_attention`, a
      blocked-softmax Pallas kernel with O(seq) memory. Used only on
      single-chip runs because Mosaic kernels cannot be
      auto-partitioned by GSPMD (multi-device raises
      ``NotImplementedError``).
    * **Multi-device TPU / CPU / GPU**:
      :func:`jax.nn.dot_product_attention`. Materialises the
      ``[heads, seq, seq]`` score matrix (O(seq^2) memory), but GSPMD
      partitions it automatically so it works with any mesh topology.
      On GPU it further lowers to cuDNN Flash-SDPA when available.
      Google's Gemma uses this same approach for multi-device training.
    """
    _, _, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    if (
        _tpu_flash_attention is not None
        and jax.default_backend() == "tpu"
        and jax.device_count() == 1
    ):
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
