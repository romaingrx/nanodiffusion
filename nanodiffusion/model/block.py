import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from nanodiffusion.config import ModelConfig
from nanodiffusion.model.attention import SelfAttention
from nanodiffusion.types import PRNGKeyArray


class AdaLNModulation(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, hidden_dim: int, *, key: PRNGKeyArray) -> None:
        linear = eqx.nn.Linear(hidden_dim, 6 * hidden_dim, use_bias=True, key=key)
        bias = linear.bias
        if bias is None:
            msg = "Linear layer must have bias"
            raise TypeError(msg)
        self.linear = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            linear,
            (jnp.zeros_like(linear.weight), jnp.zeros_like(bias)),
        )

    def __call__(
        self, cond: Float[Array, " dim"]
    ) -> tuple[
        Float[Array, " dim"],
        Float[Array, " dim"],
        Float[Array, " dim"],
        Float[Array, " dim"],
        Float[Array, " dim"],
        Float[Array, " dim"],
    ]:
        out = self.linear(jax.nn.silu(cond))
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            jnp.split(out, 6)
        )
        return shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn


class FeedForward(eqx.Module):
    gate_proj: eqx.nn.Linear
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(self, hidden_dim: int, ffn_dim: int, *, key: PRNGKeyArray) -> None:
        gkey, ukey, dkey = jax.random.split(key, 3)

        self.gate_proj = eqx.nn.Linear(hidden_dim, ffn_dim, use_bias=False, key=gkey)
        self.up_proj = eqx.nn.Linear(hidden_dim, ffn_dim, use_bias=False, key=ukey)

        down_proj = eqx.nn.Linear(ffn_dim, hidden_dim, use_bias=False, key=dkey)
        self.down_proj = eqx.tree_at(
            lambda m: m.weight, down_proj, jnp.zeros_like(down_proj.weight)
        )

    def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
        gate = jax.vmap(self.gate_proj)(x)
        up = jax.vmap(self.up_proj)(x)
        return jax.vmap(self.down_proj)(jax.nn.silu(gate) * up)


class TransformerBlock(eqx.Module):
    attn: SelfAttention
    ffn: FeedForward
    attn_norm: eqx.nn.RMSNorm
    ffn_norm: eqx.nn.RMSNorm
    adaln: AdaLNModulation

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray) -> None:
        akey, fkey, mkey = jax.random.split(key, 3)

        self.attn = SelfAttention(config.hidden_dim, config.num_heads, key=akey)
        self.ffn = FeedForward(config.hidden_dim, config.ffn_dim, key=fkey)
        self.attn_norm = eqx.nn.RMSNorm(
            config.hidden_dim, use_weight=False, use_bias=False
        )
        self.ffn_norm = eqx.nn.RMSNorm(
            config.hidden_dim, use_weight=False, use_bias=False
        )
        self.adaln = AdaLNModulation(config.hidden_dim, key=mkey)

    def __call__(
        self, x: Float[Array, "seq dim"], cond: Float[Array, " dim"]
    ) -> Float[Array, "seq dim"]:
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.adaln(
            cond
        )

        h = jax.vmap(self.attn_norm)(x)
        h = (1 + scale_attn) * h + shift_attn
        h = self.attn(h)
        x = x + gate_attn * h

        h = jax.vmap(self.ffn_norm)(x)
        h = (1 + scale_ffn) * h + shift_ffn
        h = self.ffn(h)

        return x + gate_ffn * h
