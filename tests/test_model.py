import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from nanodiffusion.config import ModelConfig
from nanodiffusion.model.attention import SelfAttention
from nanodiffusion.model.block import FeedForward, TransformerBlock
from nanodiffusion.model.embedding import TimeEmbedding, TokenEmbedding
from nanodiffusion.model.transformer import Transformer


def _activate_zeros[M: eqx.Module](model: M, key: jax.Array) -> M:
    """Replace zero-initialized weights with small random values."""

    def f(leaf: jax.Array) -> jax.Array:
        nonlocal key
        if eqx.is_array(leaf) and leaf.size > 0 and jnp.all(leaf == 0):
            key, sk = jax.random.split(key)
            return jax.random.normal(sk, leaf.shape) * 0.02
        return leaf

    return jax.tree.map(f, model, is_leaf=eqx.is_array)


@pytest.fixture
def small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        num_layers=2,
        hidden_dim=64,
        num_heads=4,
        max_seq_len=32,
    )


@pytest.fixture
def key() -> jax.Array:
    return jax.random.PRNGKey(0)


# --- Token embedding ---


def test_token_embedding_shape(key: jax.Array) -> None:
    embed = TokenEmbedding(256, 64, key=key)
    tokens = jnp.arange(16)
    out = embed(tokens)
    assert out.shape == (16, 64)


# --- Time embedding ---


def test_time_embedding_shape(key: jax.Array) -> None:
    time_embed = TimeEmbedding(64, key=key)
    out = time_embed(jnp.array(0.5))
    assert out.shape == (64,)


def test_time_embedding_different_t_gives_different_output(key: jax.Array) -> None:
    time_embed = TimeEmbedding(64, key=key)
    out_a = time_embed(jnp.array(0.1))
    out_b = time_embed(jnp.array(0.9))
    assert not jnp.allclose(out_a, out_b)


# --- Self attention ---


def test_self_attention_shape(key: jax.Array) -> None:
    attn = SelfAttention(64, 4, key=key)
    x = jax.random.normal(key, (16, 64))
    out = attn(x)
    assert out.shape == (16, 64)


def test_self_attention_is_bidirectional(key: jax.Array) -> None:
    k1, k2 = jax.random.split(key)
    attn = _activate_zeros(SelfAttention(64, 4, key=k1), k2)
    x = jax.random.normal(key, (16, 64))

    x_modified = x.at[0].set(x[0] + 10.0)
    out_orig = attn(x)
    out_modified = attn(x_modified)

    # Last position should see the change at position 0
    assert not jnp.allclose(out_orig[-1], out_modified[-1])
    # Position 0 should see changes at last position too
    x_modified_last = x.at[-1].set(x[-1] + 10.0)
    out_modified_last = attn(x_modified_last)
    assert not jnp.allclose(out_orig[0], out_modified_last[0])


# --- Feed forward ---


def test_feed_forward_shape(key: jax.Array, small_config: ModelConfig) -> None:
    ffn = FeedForward(64, small_config.ffn_dim, key=key)
    x = jax.random.normal(key, (16, 64))
    out = ffn(x)
    assert out.shape == (16, 64)


# --- Transformer block ---


def test_transformer_block_shape(
    key: jax.Array, small_config: ModelConfig
) -> None:
    block = TransformerBlock(small_config, key=key)
    x = jax.random.normal(key, (16, 64))
    cond = jax.random.normal(key, (64,))
    out = block(x, cond)
    assert out.shape == (16, 64)


def test_transformer_block_residual_at_init(
    key: jax.Array, small_config: ModelConfig
) -> None:
    block = TransformerBlock(small_config, key=key)
    x = jax.random.normal(key, (16, 64))
    cond = jax.random.normal(key, (64,))
    out = block(x, cond)
    # Zero-init AdaLN gates mean the block starts as identity
    assert jnp.allclose(out, x, atol=1e-5)


# --- Full transformer ---


def test_transformer_forward_shape(
    key: jax.Array, small_config: ModelConfig
) -> None:
    model = Transformer(small_config, key=key)
    tokens = jnp.arange(16)
    t = jnp.array(0.5)
    logits = model(tokens, t)
    assert logits.shape == (16, 256)


def test_transformer_is_bidirectional(
    key: jax.Array, small_config: ModelConfig
) -> None:
    k1, k2 = jax.random.split(key)
    model = _activate_zeros(Transformer(small_config, key=k1), k2)
    tokens_a = jnp.arange(16)
    tokens_b = tokens_a.at[0].set(100)
    t = jnp.array(0.5)

    logits_a = model(tokens_a, t)
    logits_b = model(tokens_b, t)
    # Last position should see the change at position 0
    assert not jnp.allclose(logits_a[-1], logits_b[-1])


def test_transformer_time_sensitivity(
    key: jax.Array, small_config: ModelConfig
) -> None:
    k1, k2 = jax.random.split(key)
    model = _activate_zeros(Transformer(small_config, key=k1), k2)
    tokens = jnp.arange(16)
    logits_early = model(tokens, jnp.array(0.1))
    logits_late = model(tokens, jnp.array(0.9))
    assert not jnp.allclose(logits_early, logits_late)


def test_transformer_gradient_flow(
    key: jax.Array, small_config: ModelConfig
) -> None:
    model = Transformer(small_config, key=key)
    tokens = jnp.arange(16)
    t = jnp.array(0.5)

    @eqx.filter_grad
    def compute_grad(m: Transformer) -> jax.Array:
        return jnp.sum(m(tokens, t))

    grads = compute_grad(model)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(leaf != 0) for leaf in leaves)


def test_transformer_batched_via_vmap(
    key: jax.Array, small_config: ModelConfig
) -> None:
    model = Transformer(small_config, key=key)
    batch_tokens = jnp.stack([jnp.arange(16), jnp.arange(16) + 1])
    batch_t = jnp.array([0.3, 0.7])

    batched_logits = jax.vmap(model)(batch_tokens, batch_t)
    assert batched_logits.shape == (2, 16, 256)


def test_transformer_jit_compatible(
    key: jax.Array, small_config: ModelConfig
) -> None:
    model = Transformer(small_config, key=key)
    tokens = jnp.arange(16)
    t = jnp.array(0.5)

    eager_out = model(tokens, t)
    jitted = eqx.filter_jit(model)
    jit_out = jitted(tokens, t)
    assert jnp.allclose(eager_out, jit_out, atol=1e-5)
