"""Pure generation entry points for the serve layer.

Both callables are synchronous; the FastAPI layer offloads them via
``asyncio.to_thread`` so the event loop stays responsive during XLA
blocking. Unit tests target this module directly and skip HTTP entirely.
"""

import secrets
from collections.abc import Iterator

import jax
import jax.numpy as jnp

from nanodiffusion import sampler
from nanodiffusion.chat import Conversation, render_for_completion
from nanodiffusion.config import SampleConfig
from nanodiffusion.inference import Runtime
from nanodiffusion.serve.protocol import ChatRequest, ChatResponse, StreamFrame
from nanodiffusion.tokenizer import Tokenizer
from nanodiffusion.types import PRNGKeyArray, Tokens


def _resolve(req: ChatRequest, defaults: SampleConfig) -> SampleConfig:
    return defaults.model_copy(
        update={
            f: v
            for f in SampleConfig.model_fields
            if (v := getattr(req, f)) is not None
        }
    )


def _make_key(seed: int | None) -> PRNGKeyArray:
    chosen = seed if seed is not None else secrets.randbits(32)
    return jax.random.PRNGKey(chosen)


def _prepare_prompt(
    tok: Tokenizer,
    req: ChatRequest,
    resolved: SampleConfig,
    max_seq_len: int,
) -> list[int]:
    conversation: Conversation = {"messages": req.messages}
    prompt_ids = render_for_completion(tok, conversation)

    if resolved.max_length > max_seq_len:
        msg = (
            f"max_length ({resolved.max_length}) exceeds model "
            f"max_seq_len ({max_seq_len})"
        )
        raise ValueError(msg)
    if len(prompt_ids) >= resolved.max_length:
        msg = (
            f"prompt length ({len(prompt_ids)}) must be below "
            f"max_length ({resolved.max_length})"
        )
        raise ValueError(msg)
    return prompt_ids


def generate_blocking(runtime: Runtime, req: ChatRequest) -> ChatResponse:
    resolved = _resolve(req, runtime.defaults)
    prompt_ids = _prepare_prompt(runtime.tok, req, resolved, runtime.max_seq_len)
    prompt_tokens = jnp.array(prompt_ids)

    tokens = sampler.sample_tokens(
        runtime.model,
        prompt_tokens,
        schedule=runtime.schedule,
        mask_token_id=runtime.tok.mask_token_id,
        max_length=resolved.max_length,
        steps=resolved.steps,
        temperature=resolved.temperature,
        top_k=resolved.top_k,
        top_p=resolved.top_p,
        key=_make_key(req.seed),
    )
    token_list = tokens.tolist()
    return ChatResponse(
        text=runtime.tok.decode(token_list),
        tokens=token_list,
        prompt_len=len(prompt_ids),
    )


def generate_stream(runtime: Runtime, req: ChatRequest) -> Iterator[StreamFrame]:
    """Validate eagerly, then return a lazy iterator of :class:`StreamFrame`.

    Errors raise synchronously so the HTTP handler maps them to 422
    before committing to a streaming response.
    """
    resolved = _resolve(req, runtime.defaults)
    prompt_ids = _prepare_prompt(runtime.tok, req, resolved, runtime.max_seq_len)
    prompt_tokens = jnp.array(prompt_ids)
    key = _make_key(req.seed)
    return _stream_frames(runtime, resolved, prompt_tokens, key)


def _stream_frames(
    runtime: Runtime,
    resolved: SampleConfig,
    prompt_tokens: Tokens,
    key: PRNGKeyArray,
) -> Iterator[StreamFrame]:
    mask_id = runtime.tok.mask_token_id
    for step in sampler.sample(
        runtime.model,
        prompt_tokens,
        schedule=runtime.schedule,
        mask_token_id=mask_id,
        max_length=resolved.max_length,
        steps=resolved.steps,
        temperature=resolved.temperature,
        top_k=resolved.top_k,
        top_p=resolved.top_p,
        key=key,
    ):
        token_list = step.tokens.tolist()
        yield StreamFrame(
            step=step.step,
            total=step.total_steps,
            tokens=token_list,
            text=runtime.tok.decode(token_list),
            mask_positions=[i for i, t in enumerate(token_list) if t == mask_id],
        )
