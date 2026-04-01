import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from nanodiffusion.types import PRNGKeyArray, Scalar


class TokenEmbedding(eqx.Module):
    embed: eqx.nn.Embedding

    def __init__(self, vocab_size: int, embed_dim: int, *, key: PRNGKeyArray) -> None:
        k1, k2 = jax.random.split(key)
        self.embed = eqx.nn.Embedding(vocab_size, embed_dim, key=k1)
        weight = jax.random.normal(k2, (vocab_size, embed_dim)) * 0.02
        self.embed = eqx.tree_at(lambda e: e.weight, self.embed, weight)

    def __call__(self, tokens: Int[Array, " seq"]) -> Float[Array, "seq dim"]:
        return jax.vmap(self.embed)(tokens)


def sinusoidal_embedding(t: Scalar, dim: int) -> Float[Array, " dim"]:
    half = dim // 2
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half) / half)
    args = t * freqs
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)])


class TimeEmbedding(eqx.Module):
    linear_1: eqx.nn.Linear
    linear_2: eqx.nn.Linear
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, *, key: PRNGKeyArray) -> None:
        self.dim = dim
        k1, k2 = jax.random.split(key)
        self.linear_1 = eqx.nn.Linear(dim, dim, use_bias=True, key=k1)
        self.linear_2 = eqx.nn.Linear(dim, dim, use_bias=True, key=k2)

    def __call__(self, t: Scalar) -> Float[Array, " dim"]:
        x = sinusoidal_embedding(t, self.dim)
        x = self.linear_1(x)
        x = jax.nn.silu(x)
        return self.linear_2(x)
