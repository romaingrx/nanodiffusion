import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from nanodiffusion.ops import attention
from nanodiffusion.types import PRNGKeyArray


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

        # RMSNorm + RoPE are per-head; pivot to (N, T, H) so each head's
        # (T, H) slice can be normed and rotated independently, then stay
        # in that layout because :func:`nanodiffusion.ops.attention`
        # takes the same layout.
        q = jnp.transpose(q, (1, 0, 2))
        k = jnp.transpose(k, (1, 0, 2))
        v = jnp.transpose(v, (1, 0, 2))
        q = jax.vmap(jax.vmap(self.q_norm))(q)
        k = jax.vmap(jax.vmap(self.k_norm))(k)
        q = jax.vmap(self.rope)(q)
        k = jax.vmap(self.rope)(k)

        out = attention(q, k, v)

        out = jnp.transpose(out, (1, 0, 2))
        out = out.reshape(seq_len, self.num_heads * self.head_dim)
        return jax.vmap(self.o_proj)(out)
