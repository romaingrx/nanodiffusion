import equinox as eqx
import jax

from nanodiffusion.config import ModelConfig
from nanodiffusion.model._base import DiffusionModel
from nanodiffusion.model.block import TransformerBlock
from nanodiffusion.model.embedding import TimeEmbedding, TokenEmbedding
from nanodiffusion.types import Logits, PRNGKeyArray, Scalar, Tokens


class Transformer(DiffusionModel):
    embed: TokenEmbedding
    time_embed: TimeEmbedding
    blocks: list[TransformerBlock]
    final_norm: eqx.nn.RMSNorm
    lm_head: eqx.nn.Linear

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray) -> None:
        keys = jax.random.split(key, config.num_layers + 3)

        self.embed = TokenEmbedding(config.vocab_size, config.hidden_dim, key=keys[0])
        self.time_embed = TimeEmbedding(config.hidden_dim, key=keys[1])
        self.blocks = [
            TransformerBlock(config, key=keys[2 + i]) for i in range(config.num_layers)
        ]
        self.final_norm = eqx.nn.RMSNorm(
            config.hidden_dim, use_weight=False, use_bias=False
        )

        self.lm_head = eqx.nn.Linear(
            config.hidden_dim, config.vocab_size, use_bias=False, key=keys[-1]
        )

    def __call__(self, tokens: Tokens, t: Scalar) -> Logits:
        x = self.embed(tokens)
        cond = self.time_embed(t)

        for block in self.blocks:
            x = block(x, cond)

        x = jax.vmap(self.final_norm)(x)
        return jax.vmap(self.lm_head)(x)
