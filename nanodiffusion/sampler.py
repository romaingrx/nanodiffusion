"""Iterative unmasking sampler for masked discrete diffusion.

Implements the MDLM ancestral sampler (Sahoo et al., NeurIPS 2024) with
nucleus sampling support (critical for text quality per ReMDM findings).
"""

import functools
from collections.abc import Iterator
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from nanodiffusion.model import DiffusionModel
from nanodiffusion.schedule import NoiseSchedule, mask_chance
from nanodiffusion.types import PRNGKeyArray, Prompt, Scalar, Tokens


class SampleStep(NamedTuple):
    tokens: Tokens
    step: int
    total_steps: int


def top_k_filtering(logits: Float[Array, " vocab"], k: int) -> Float[Array, " vocab"]:
    if k <= 0:
        return logits
    top_values, _ = jax.lax.top_k(logits, k)
    threshold = top_values[-1]
    return jnp.where(logits >= threshold, logits, -jnp.inf)


def top_p_filtering(logits: Float[Array, " vocab"], p: float) -> Float[Array, " vocab"]:
    if p >= 1.0:
        return logits
    sorted_indices = jnp.argsort(-logits)
    sorted_logits = logits[sorted_indices]
    sorted_probs = jax.nn.softmax(sorted_logits)
    # Shift cumsum right by 1 so we include the token that crosses the threshold
    cumulative_probs = jnp.cumsum(sorted_probs) - sorted_probs
    cutoff = jnp.where(cumulative_probs < p, sorted_logits, -jnp.inf)
    return jnp.empty_like(logits).at[sorted_indices].set(cutoff)


def filter_logits(
    logits: Float[Array, " vocab"],
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Float[Array, " vocab"]:
    logits = top_k_filtering(logits, top_k)
    logits = top_p_filtering(logits, top_p)
    return logits / temperature


def _reverse_posterior(
    filtered_logits: Float[Array, "seq vocab"],
    move_chance_t: Scalar,
    move_chance_s: Scalar,
    mask_token_id: int,
) -> Float[Array, "seq vocab"]:
    """Log-space reverse posterior q(x_s | x_t, x0_hat). See MDLM Eq. 8."""
    log_p_x0 = jax.nn.log_softmax(filtered_logits, axis=-1)
    log_unmask = jnp.log(move_chance_t - move_chance_s)
    log_q = log_p_x0 + log_unmask
    log_stay = jnp.log(move_chance_s)
    return log_q.at[:, mask_token_id].set(log_stay)


_T_MIN = 1e-5


def sample(
    model: DiffusionModel,
    prompt_tokens: Prompt,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    max_length: int,
    steps: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    key: PRNGKeyArray,
) -> Iterator[SampleStep]:
    """MDLM ancestral sampler yielding each denoising step.

    Ref: Sahoo et al., "Simple and Effective Masked Diffusion Language
    Models", NeurIPS 2024, Algorithm 1 (ddpm_cache variant).
    """
    prompt_len = prompt_tokens.shape[0]
    if max_length <= prompt_len:
        msg = f"max_length ({max_length}) must exceed prompt length ({prompt_len})"
        raise ValueError(msg)
    gen_len = max_length - prompt_len
    x = jnp.concatenate([prompt_tokens, jnp.full(gen_len, mask_token_id)])
    is_prompt = jnp.concatenate(
        [
            jnp.ones(prompt_len, dtype=bool),
            jnp.zeros(gen_len, dtype=bool),
        ]
    )

    timesteps = jnp.linspace(1.0, _T_MIN, steps + 1)

    _filter = functools.partial(
        filter_logits, temperature=temperature, top_k=top_k, top_p=top_p
    )

    for i in range(steps):
        t = timesteps[i]
        s = timesteps[i + 1]

        move_t = mask_chance(schedule, t)
        move_s = mask_chance(schedule, s)

        logits = model(x, t)
        logits = logits.at[:, mask_token_id].set(-1e9)
        filtered = jax.vmap(_filter)(logits)

        log_q = _reverse_posterior(filtered, move_t, move_s, mask_token_id)

        key, step_key = jax.random.split(key)
        x_new = jax.random.categorical(step_key, log_q, axis=-1)

        is_masked = x == mask_token_id
        x = jnp.where(is_masked & ~is_prompt, x_new, x)

        yield SampleStep(x, i, steps)

    # Final denoising: argmax remaining masks
    logits = model(x, jnp.array(_T_MIN))
    logits = logits.at[:, mask_token_id].set(-1e9)
    x = jnp.where(x == mask_token_id, jnp.argmax(logits, axis=-1), x)

    yield SampleStep(x, steps, steps)


def sample_tokens(
    model: DiffusionModel,
    prompt_tokens: Prompt,
    *,
    schedule: NoiseSchedule,
    mask_token_id: int,
    max_length: int,
    steps: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    key: PRNGKeyArray,
) -> Tokens:
    """Run the sampler and return the final token sequence."""
    result: Tokens = prompt_tokens
    for step in sample(
        model,
        prompt_tokens,
        schedule=schedule,
        mask_token_id=mask_token_id,
        max_length=max_length,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        key=key,
    ):
        result = step.tokens
    return result
