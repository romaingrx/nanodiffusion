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

import functools
import math
from collections.abc import Callable

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

try:
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        FullMask,
        MultiHeadMask,
        make_splash_mha_single_device,
    )
except ImportError:
    make_splash_mha_single_device = None
    FullMask = None
    MultiHeadMask = None


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

    * **TPU** (single- or multi-device):
      :func:`jax.experimental.pallas.ops.tpu.flash_attention`, a
      blocked-softmax Pallas kernel with O(seq) memory. On multi-device
      TPU, ``compute_loss`` wraps the model call in ``shard_map`` so
      GSPMD never tries to auto-partition the Pallas custom call.
    * **CPU / GPU**:
      :func:`jax.nn.dot_product_attention`. Materialises the
      ``[heads, seq, seq]`` score matrix (O(seq^2) memory).
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


type _HeadsTensor = Float[Array, "heads seq head_dim"]
type _SplashKernelFn = Callable[
    [_HeadsTensor, _HeadsTensor, _HeadsTensor],
    _HeadsTensor,
]


@functools.lru_cache(maxsize=8)
def _splash_kernel(num_heads: int, seq_len: int) -> _SplashKernelFn:
    """Build and cache a SplashAttention kernel for the given geometry.

    Kernel construction allocates host-side block structures that cannot
    be created inside ``jax.jit``. Caching by ``(num_heads, seq_len)``
    ensures at most one kernel per distinct shape seen during training.
    """
    if (
        make_splash_mha_single_device is None
        or FullMask is None
        or MultiHeadMask is None
    ):
        msg = "SplashAttention requires jax[tpu] with Pallas Mosaic support"
        raise ImportError(msg)
    mask = MultiHeadMask(masks=[FullMask((seq_len, seq_len)) for _ in range(num_heads)])
    return make_splash_mha_single_device(mask, head_shards=1, q_seq_shards=1)  # pyright: ignore[reportReturnType]


def splash_attention(
    q: Float[Array, "heads seq head_dim"],
    k: Float[Array, "heads seq head_dim"],
    v: Float[Array, "heads seq head_dim"],
) -> Float[Array, "heads seq head_dim"]:
    """SplashAttention for use inside ``shard_map`` on multi-device TPU.

    Each device invokes a single-device Pallas Mosaic kernel with O(seq)
    memory via :func:`make_splash_mha_single_device` and a
    :class:`FullMask` (non-causal, bidirectional). The kernel object is
    cached at the module level because it cannot be constructed inside
    ``jax.jit``.

    Input layout is ``(num_heads, seq_len, head_dim)`` -- the same as
    :func:`attention` -- so the two are drop-in replacements for each
    other.
    """
    num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    kernel = _splash_kernel(num_heads, seq_len)
    return kernel(q * scale, k, v)
