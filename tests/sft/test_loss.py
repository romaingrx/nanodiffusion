import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nanodiffusion.config import ModelConfig
from nanodiffusion.loss import TimeSampler, token_nll
from nanodiffusion.model.transformer import Transformer
from nanodiffusion.pretrain.loss import forward_mask, masked_nll
from nanodiffusion.schedule import LogLinearSchedule, loss_weight
from nanodiffusion.sft.loss import compute_sft_loss, sft_forward_mask
from tests._helpers import inexact_leaves


def _fixed_sampler(values: jax.Array) -> TimeSampler:
    def sampler(batch_size: int, *, key: jax.Array) -> jax.Array:  # noqa: ARG001
        assert values.shape == (batch_size,)
        return values

    return sampler


def test_token_nll_refactor_matches_original_masked_nll(key: jax.Array) -> None:
    """masked_nll re-expressed on top of token_nll must be bitwise stable.

    Regression against the Slice 1 extraction: an accidental numerical
    reshuffle would drift masked_nll's output and silently change pretrain
    loss values. We compare against a hand-rolled reference that mirrors
    the pre-refactor formula.
    """
    logits_key, x_key, mask_key = jax.random.split(key, 3)
    logits = jax.random.normal(logits_key, (8, 16))
    x0 = jax.random.randint(x_key, (8,), 0, 16)
    is_masked = jax.random.bernoulli(mask_key, p=0.5, shape=(8,))
    weight = jnp.array(1.7)

    reference = (
        weight
        * (-jax.nn.log_softmax(logits, axis=-1)[jnp.arange(8), x0] * is_masked).sum()
        / jnp.maximum(is_masked.sum(), 1)
    )
    actual = masked_nll(logits, x0, is_masked, weight)

    np.testing.assert_array_equal(actual, reference)


def test_token_nll_shape_and_sign(key: jax.Array) -> None:
    logits = jax.random.normal(key, (12, 32))
    x0 = jax.random.randint(key, (12,), 0, 32)
    nll = token_nll(logits, x0)
    assert nll.shape == (12,)
    assert jnp.all(nll >= 0)


def test_sft_forward_mask_never_masks_prompt_positions(key: jax.Array) -> None:
    """Across a range of t, loss_mask=False positions must stay clean."""
    schedule = LogLinearSchedule()
    x0 = jnp.arange(32, dtype=jnp.int32)
    # First half is prompt, second half is response.
    loss_mask = jnp.concatenate(
        [jnp.zeros(16, dtype=jnp.bool_), jnp.ones(16, dtype=jnp.bool_)]
    )
    for t_val in [0.05, 0.25, 0.5, 0.75, 0.95]:
        t = jnp.array(t_val)
        t_key = jax.random.fold_in(key, int(t_val * 1000))
        xt, is_masked = sft_forward_mask(
            x0, loss_mask, t, schedule=schedule, mask_token_id=999, key=t_key
        )
        # Prompt half is untouched in both xt and is_masked.
        np.testing.assert_array_equal(xt[:16], x0[:16])
        assert not jnp.any(is_masked[:16])


def test_sft_forward_mask_matches_forward_mask_when_all_supervised(
    key: jax.Array,
) -> None:
    """With loss_mask=all-True, sft_forward_mask degenerates to forward_mask."""
    schedule = LogLinearSchedule()
    x0 = jnp.arange(24, dtype=jnp.int32)
    loss_mask = jnp.ones(24, dtype=jnp.bool_)
    t = jnp.array(0.4)

    xt_sft, mask_sft = sft_forward_mask(
        x0, loss_mask, t, schedule=schedule, mask_token_id=999, key=key
    )
    xt_plain, mask_plain = forward_mask(
        x0, t, schedule=schedule, mask_token_id=999, key=key
    )
    np.testing.assert_array_equal(xt_sft, xt_plain)
    np.testing.assert_array_equal(mask_sft, mask_plain)


def test_compute_sft_loss_zero_when_loss_mask_all_false(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """All-zero loss_mask means no supervision anywhere → loss must be exactly 0."""
    model_key, batch_key, loss_key = jax.random.split(key, 3)
    model = Transformer(small_config, key=model_key)
    x0 = jax.random.randint(
        batch_key, (4, small_config.max_seq_len), 0, small_config.vocab_size - 1
    )
    loss_mask = jnp.zeros((4, small_config.max_seq_len), dtype=jnp.bool_)

    loss = compute_sft_loss(
        model,
        x0,
        loss_mask,
        schedule=LogLinearSchedule(),
        mask_token_id=small_config.vocab_size - 1,
        key=loss_key,
    )
    assert float(loss) == 0.0


def test_compute_sft_loss_zero_gradient_when_loss_mask_all_false(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """No supervision → the model parameter gradient tree must be all zeros.

    This is the load-bearing 'prompt positions don't contribute' assertion
    end-to-end: if the loss is zero and the gradient pytree is also zero,
    then nothing about the model's handling of prompt positions can leak
    into an optimizer update.
    """
    model_key, batch_key, loss_key = jax.random.split(key, 3)
    model = Transformer(small_config, key=model_key)
    x0 = jax.random.randint(
        batch_key, (2, small_config.max_seq_len), 0, small_config.vocab_size - 1
    )
    loss_mask = jnp.zeros((2, small_config.max_seq_len), dtype=jnp.bool_)
    schedule = LogLinearSchedule()

    def loss_fn(m: Transformer) -> jax.Array:
        return compute_sft_loss(
            m,
            x0,
            loss_mask,
            schedule=schedule,
            mask_token_id=small_config.vocab_size - 1,
            key=loss_key,
        )

    grads = eqx.filter_grad(loss_fn)(model)
    for leaf in inexact_leaves(grads):
        np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))


def test_compute_sft_loss_equals_reference_global_aggregation(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """Against a hand-rolled reference using the same schedule + sampler.

    Pins the batch-level aggregation formula: sum of per-row
    ``w(t_i) * sum(nll_i * is_masked_i)`` divided by total masked count.
    """
    model_key, batch_key, loss_key = jax.random.split(key, 3)
    model = Transformer(small_config, key=model_key)
    batch = 4
    seq = small_config.max_seq_len
    x0 = jax.random.randint(batch_key, (batch, seq), 0, small_config.vocab_size - 1)
    # Half the tokens on each row are supervised, with a different split per row.
    loss_mask = jnp.stack(
        [
            jnp.arange(seq) >= (seq // 4),
            jnp.arange(seq) < (3 * seq // 4),
            jnp.arange(seq) % 2 == 0,
            jnp.arange(seq) >= 1,
        ]
    )
    # Deterministic t_batch so both paths see identical noise rates.
    t_batch = jnp.array([0.15, 0.35, 0.55, 0.8])
    schedule = LogLinearSchedule()
    mask_id = small_config.vocab_size - 1

    loss = compute_sft_loss(
        model,
        x0,
        loss_mask,
        schedule=schedule,
        mask_token_id=mask_id,
        key=loss_key,
        sampler=_fixed_sampler(t_batch),
    )

    # Reference: compute per-row contributions then aggregate globally.
    _, t_key = jax.random.split(loss_key)
    del t_key  # compute_sft_loss splits after the sampler call; mirror exactly.
    key_after_sampler = jax.random.split(loss_key)[0]
    row_keys = jax.random.split(key_after_sampler, batch)
    total_weighted = jnp.array(0.0)
    total_count = jnp.array(0.0)
    for i in range(batch):
        xi = x0[i]
        li = loss_mask[i]
        ti = t_batch[i]
        ki = row_keys[i]
        xt, is_masked = sft_forward_mask(
            xi, li, ti, schedule=schedule, mask_token_id=mask_id, key=ki
        )
        logits = model(xt, ti)
        nll_i = (token_nll(logits, xi) * is_masked).sum()
        w_i = loss_weight(schedule, ti)
        total_weighted = total_weighted + w_i * nll_i
        total_count = total_count + is_masked.sum().astype(jnp.float32)
    reference = total_weighted / jnp.maximum(total_count, 1.0)

    np.testing.assert_allclose(float(loss), float(reference), rtol=1e-5, atol=1e-6)


def test_compute_sft_loss_ignores_nonmasked_rows_in_denominator(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """A fully-prompt row must contribute 0 to both numerator and denominator.

    So a mixed batch with some rows having supervision and others with
    none must produce the same loss as the sub-batch containing only the
    supervised rows.
    """
    model_key, batch_key, loss_key = jax.random.split(key, 3)
    model = Transformer(small_config, key=model_key)
    seq = small_config.max_seq_len
    mask_id = small_config.vocab_size - 1
    schedule = LogLinearSchedule()

    supervised_row = jax.random.randint(batch_key, (seq,), 0, mask_id)
    prompt_row = jax.random.randint(
        jax.random.fold_in(batch_key, 1), (seq,), 0, mask_id
    )
    full_mask = jnp.ones((1, seq), dtype=jnp.bool_)
    empty_mask = jnp.zeros((1, seq), dtype=jnp.bool_)

    sampler = _fixed_sampler(jnp.array([0.5]))
    loss_single = compute_sft_loss(
        model,
        supervised_row[None],
        full_mask,
        schedule=schedule,
        mask_token_id=mask_id,
        key=loss_key,
        sampler=sampler,
    )

    sampler_pair = _fixed_sampler(jnp.array([0.5, 0.5]))
    loss_pair = compute_sft_loss(
        model,
        jnp.stack([supervised_row, prompt_row]),
        jnp.concatenate([full_mask, empty_mask]),
        schedule=schedule,
        mask_token_id=mask_id,
        key=loss_key,
        sampler=sampler_pair,
    )

    # The second row contributes nothing via the sampler's fixed t=0.5
    # and the empty loss_mask, so the global aggregation must match the
    # single-row computation exactly.
    np.testing.assert_allclose(float(loss_pair), float(loss_single), rtol=1e-5)
