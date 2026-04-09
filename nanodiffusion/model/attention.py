import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from nanodiffusion.types import PRNGKeyArray

# Pallas Flash Attention on TPU avoids materialising the [N, T, T] score
# matrix that XLA's ``jax.nn.dot_product_attention`` pattern still emits
# as a plain ``dot_general`` on TPU. The import is safe on non-TPU
# backends (the call is what fails) so we keep the reference and gate
# usage on the runtime backend.
try:
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        flash_attention as _tpu_flash_attention,
    )
except ImportError:
    _tpu_flash_attention = None


class SelfAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    q_norm: eqx.nn.RMSNorm
    k_norm: eqx.nn.RMSNorm
    rope: eqx.nn.RotaryPositionalEmbedding
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, hidden_dim: int, num_heads: int, *, key: PRNGKeyArray) -> None:
        qkey, kkey, vkey, okey = jax.random.split(key, 4)

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=qkey)
        self.k_proj = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=kkey)
        self.v_proj = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=vkey)

        self.o_proj = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=okey)

        self.q_norm = eqx.nn.RMSNorm(self.head_dim, use_weight=False, use_bias=False)
        self.k_norm = eqx.nn.RMSNorm(self.head_dim, use_weight=False, use_bias=False)
        self.rope = eqx.nn.RotaryPositionalEmbedding(self.head_dim)

    def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
        q = jax.vmap(self.q_proj)(x)
        k = jax.vmap(self.k_proj)(x)
        v = jax.vmap(self.v_proj)(x)

        seq_len = x.shape[0]
        q = q.reshape(seq_len, self.num_heads, self.head_dim)
        k = k.reshape(seq_len, self.num_heads, self.head_dim)
        v = v.reshape(seq_len, self.num_heads, self.head_dim)

        # RMSNorm + RoPE are defined per head, so briefly pivot to (N, T, H).
        q = jnp.transpose(q, (1, 0, 2))
        k = jnp.transpose(k, (1, 0, 2))
        q = jax.vmap(jax.vmap(self.q_norm))(q)
        k = jax.vmap(jax.vmap(self.k_norm))(k)
        q = jax.vmap(self.rope)(q)
        k = jax.vmap(self.rope)(k)

        scale = 1.0 / math.sqrt(self.head_dim)

        if _tpu_flash_attention is not None and jax.default_backend() == "tpu":
            # Pallas Flash Attention: blocked kernel, O(T) memory, never
            # materialises the [N, T, T] score matrix. Expects
            # (B, N, T, H); we already have q, k in (N, T, H), just
            # pivot v and add a unit batch dim.
            v_nth = jnp.transpose(v, (1, 0, 2))
            out = _tpu_flash_attention(
                q[jnp.newaxis],
                k[jnp.newaxis],
                v_nth[jnp.newaxis],
                sm_scale=scale,
            )[0]  # (N, T, H)
            out = jnp.transpose(out, (1, 0, 2))  # (T, N, H)
        else:
            # Portable fallback: jax.nn.dot_product_attention wants
            # (T, N, H). On TPU XLA lowers this to a plain dot_general
            # that keeps the score matrix alive, so it is strictly
            # worse than the Pallas path; we only hit this branch on
            # CPU and GPU.
            q_tnh = jnp.transpose(q, (1, 0, 2))
            k_tnh = jnp.transpose(k, (1, 0, 2))
            out = jax.nn.dot_product_attention(q_tnh, k_tnh, v, scale=scale)

        out = out.reshape(seq_len, self.num_heads * self.head_dim)
        return jax.vmap(self.o_proj)(out)
