import itertools

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from nanodiffusion.model.transformer import Transformer
from nanodiffusion.sampler import (
    filter_logits,
    sample,
    sample_tokens,
    top_k_filtering,
    top_p_filtering,
)
from nanodiffusion.schedule import CosineSchedule, LogLinearSchedule, NoiseSchedule

MASK_ID = 255
SEQ_LEN = 32


@pytest.fixture(params=[LogLinearSchedule(), CosineSchedule()])
def schedule(request: pytest.FixtureRequest) -> NoiseSchedule:
    return request.param


class TestTopKFiltering:
    def test_keeps_exactly_k(self) -> None:
        logits = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = top_k_filtering(logits, k=3)
        finite = jnp.isfinite(result)
        assert int(finite.sum()) == 3
        assert jnp.all(result[finite] == logits[finite])

    def test_noop_when_k_zero(self) -> None:
        logits = jnp.array([1.0, 2.0, 3.0])
        result = top_k_filtering(logits, k=0)
        assert jnp.allclose(result, logits)


class TestTopPFiltering:
    def test_keeps_nucleus(self) -> None:
        # Probabilities after softmax: roughly [0.09, 0.67, 0.24]
        logits = jnp.array([0.0, 2.0, 1.0])
        result = top_p_filtering(logits, p=0.9)
        finite = jnp.isfinite(result)
        # Top-2 tokens (indices 1, 2) cover ~91% → should keep 2
        assert int(finite.sum()) == 2

    def test_noop_when_p_one(self) -> None:
        logits = jnp.array([1.0, 2.0, 3.0])
        result = top_p_filtering(logits, p=1.0)
        assert jnp.allclose(result, logits)

    def test_always_keeps_top1(self) -> None:
        # Even with very small p, top-1 must survive
        logits = jnp.array([0.0, 10.0, 0.0])
        result = top_p_filtering(logits, p=0.01)
        finite = jnp.isfinite(result)
        assert int(finite.sum()) >= 1
        assert jnp.isfinite(result[1])


class TestFilterLogits:
    def test_temperature_scaling(self) -> None:
        logits = jnp.array([2.0, 4.0, 6.0])
        result = filter_logits(logits, temperature=2.0, top_k=0, top_p=1.0)
        assert jnp.allclose(result, logits / 2.0)

    def test_top_k_then_top_p(self) -> None:
        logits = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = filter_logits(logits, temperature=1.0, top_k=3, top_p=0.9)
        finite = jnp.isfinite(result)
        # top-k=3 keeps indices {1,2,4}, then top-p further narrows
        assert int(finite.sum()) <= 3
        assert int(finite.sum()) >= 1


class TestSampler:
    def test_mask_count_decreases_monotonically(
        self, model: Transformer, key: jax.Array, schedule: NoiseSchedule
    ) -> None:
        prompt = jnp.array([0, 1, 2])
        counts = [
            int((step.tokens == MASK_ID).sum())
            for step in sample(
                model,
                prompt,
                schedule=schedule,
                mask_token_id=MASK_ID,
                max_length=SEQ_LEN,
                steps=8,
                key=key,
            )
        ]
        for a, b in itertools.pairwise(counts):
            assert a >= b

    def test_final_output_no_masks(
        self, model: Transformer, key: jax.Array, schedule: NoiseSchedule
    ) -> None:
        prompt = jnp.array([0, 1, 2])
        tokens = sample_tokens(
            model,
            prompt,
            schedule=schedule,
            mask_token_id=MASK_ID,
            max_length=SEQ_LEN,
            steps=8,
            key=key,
        )
        assert int((tokens == MASK_ID).sum()) == 0

    def test_prompt_preserved(
        self, model: Transformer, key: jax.Array, schedule: NoiseSchedule
    ) -> None:
        prompt = jnp.array([0, 1, 2])
        for step in sample(
            model,
            prompt,
            schedule=schedule,
            mask_token_id=MASK_ID,
            max_length=SEQ_LEN,
            steps=8,
            key=key,
        ):
            assert jnp.array_equal(step.tokens[: len(prompt)], prompt)

    def test_deterministic_same_key(
        self, model: Transformer, key: jax.Array, schedule: NoiseSchedule
    ) -> None:
        prompt = jnp.array([0, 1, 2])
        kwargs = {
            "schedule": schedule,
            "mask_token_id": MASK_ID,
            "max_length": SEQ_LEN,
            "steps": 4,
            "key": key,
        }
        a = sample_tokens(model, prompt, **kwargs)
        b = sample_tokens(model, prompt, **kwargs)
        assert jnp.array_equal(a, b)

    def test_different_key_different_output(
        self, model: Transformer, schedule: NoiseSchedule
    ) -> None:
        prompt = jnp.array([0, 1, 2])
        k1, k2 = jax.random.split(jax.random.PRNGKey(99))
        kwargs = {
            "schedule": schedule,
            "mask_token_id": MASK_ID,
            "max_length": SEQ_LEN,
            "steps": 8,
        }
        a = sample_tokens(model, prompt, **kwargs, key=k1)
        b = sample_tokens(model, prompt, **kwargs, key=k2)
        assert not jnp.array_equal(a, b)

    def test_yield_count(
        self, model: Transformer, key: jax.Array, schedule: NoiseSchedule
    ) -> None:
        prompt = jnp.array([0, 1, 2])
        steps_list = list(
            sample(
                model,
                prompt,
                schedule=schedule,
                mask_token_id=MASK_ID,
                max_length=SEQ_LEN,
                steps=8,
                key=key,
            )
        )
        assert len(steps_list) == 9  # steps + 1

    def test_sample_tokens_matches_last_yield(
        self, model: Transformer, key: jax.Array, schedule: NoiseSchedule
    ) -> None:
        prompt = jnp.array([0, 1, 2])
        kwargs = {
            "schedule": schedule,
            "mask_token_id": MASK_ID,
            "max_length": SEQ_LEN,
            "steps": 4,
            "key": key,
        }
        steps_list = list(sample(model, prompt, **kwargs))
        tokens = sample_tokens(model, prompt, **kwargs)
        assert jnp.array_equal(steps_list[-1].tokens, tokens)

    def test_single_step(
        self, model: Transformer, key: jax.Array, schedule: NoiseSchedule
    ) -> None:
        prompt = jnp.array([0, 1, 2])
        tokens = sample_tokens(
            model,
            prompt,
            schedule=schedule,
            mask_token_id=MASK_ID,
            max_length=SEQ_LEN,
            steps=1,
            key=key,
        )
        assert tokens.shape == (SEQ_LEN,)
        assert int((tokens == MASK_ID).sum()) == 0

    def test_minimal_prompt(
        self, model: Transformer, key: jax.Array, schedule: NoiseSchedule
    ) -> None:
        prompt = jnp.array([0])
        tokens = sample_tokens(
            model,
            prompt,
            schedule=schedule,
            mask_token_id=MASK_ID,
            max_length=SEQ_LEN,
            steps=4,
            key=key,
        )
        assert tokens.shape == (SEQ_LEN,)
        assert tokens[0] == 0

    def test_rejects_max_length_too_short(
        self, model: Transformer, key: jax.Array, schedule: NoiseSchedule
    ) -> None:
        prompt = jnp.array([0, 1, 2])
        with pytest.raises(ValueError, match=r"max_length.*must exceed prompt length"):
            sample_tokens(
                model,
                prompt,
                schedule=schedule,
                mask_token_id=MASK_ID,
                max_length=3,
                steps=4,
                key=key,
            )

    def test_rejects_2d_prompt(
        self, model: Transformer, key: jax.Array, schedule: NoiseSchedule
    ) -> None:
        bad_prompt = jnp.array([[0, 1, 2]])  # 2D — should be 1D
        with pytest.raises(TypeCheckError):
            sample_tokens(
                model,
                bad_prompt,
                schedule=schedule,
                mask_token_id=MASK_ID,
                max_length=SEQ_LEN,
                steps=1,
                key=key,
            )

    def test_rejects_float_prompt(
        self, model: Transformer, key: jax.Array, schedule: NoiseSchedule
    ) -> None:
        bad_prompt = jnp.array([0.0, 1.0, 2.0])  # float — should be int
        with pytest.raises(TypeCheckError):
            sample_tokens(
                model,
                bad_prompt,
                schedule=schedule,
                mask_token_id=MASK_ID,
                max_length=SEQ_LEN,
                steps=1,
                key=key,
            )
