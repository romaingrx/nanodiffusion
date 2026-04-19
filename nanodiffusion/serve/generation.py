"""Pure generation entry points for the serve layer.

Both callables are synchronous; the FastAPI layer offloads them via
``asyncio.to_thread`` so the event loop stays responsive during XLA
blocking. Unit tests target this module directly and skip HTTP entirely.
"""

import dataclasses
import secrets
from collections.abc import Iterator

import jax
import jax.numpy as jnp

from nanodiffusion import sampler
from nanodiffusion.chat import (
    Conversation,
    render_for_completion,
)
from nanodiffusion.chat import (
    Message as ChatMessage,
)
from nanodiffusion.serve.protocol import (
    ChatRequest,
    ChatResponse,
    SampleDefaults,
    StreamFrame,
)
from nanodiffusion.serve.runtime import Runtime
from nanodiffusion.tokenizer import SpecialToken, Tokenizer
from nanodiffusion.types import PRNGKeyArray, Tokens


@dataclasses.dataclass(frozen=True)
class _Resolved:
    steps: int
    temperature: float
    top_k: int
    top_p: float
    max_length: int


def _resolve(req: ChatRequest, defaults: SampleDefaults) -> _Resolved:
    return _Resolved(
        steps=req.steps if req.steps is not None else defaults.steps,
        temperature=req.temperature
        if req.temperature is not None
        else defaults.temperature,
        top_k=req.top_k if req.top_k is not None else defaults.top_k,
        top_p=req.top_p if req.top_p is not None else defaults.top_p,
        max_length=req.max_length
        if req.max_length is not None
        else defaults.max_length,
    )


def _make_key(seed: int | None) -> PRNGKeyArray:
    chosen = seed if seed is not None else secrets.randbits(32)
    return jax.random.PRNGKey(chosen)


def _render_with_blocks(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids).replace(SpecialToken.MASK.value, "███")


def _mask_positions(tokens: Tokens, mask_id: int) -> list[int]:
    return jnp.where(tokens == mask_id)[0].tolist()


def _prepare_prompt(
    tok: Tokenizer,
    messages: list[ChatMessage],
    resolved: _Resolved,
    max_seq_len: int,
) -> list[int]:
    conversation: Conversation = {"messages": messages}
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


def _to_message_list(req: ChatRequest) -> list[ChatMessage]:
    """Convert Pydantic Message list to chat.Message TypedDicts for the renderer."""
    return [{"role": m.role, "content": m.content} for m in req.messages]


def generate_blocking(runtime: Runtime, req: ChatRequest) -> ChatResponse:
    resolved = _resolve(req, runtime.defaults)
    messages = _to_message_list(req)
    prompt_ids = _prepare_prompt(runtime.tok, messages, resolved, runtime.max_seq_len)
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
        text=_render_with_blocks(runtime.tok, token_list),
        tokens=token_list,
        prompt_len=len(prompt_ids),
    )


def generate_stream(runtime: Runtime, req: ChatRequest) -> Iterator[StreamFrame]:
    resolved = _resolve(req, runtime.defaults)
    messages = _to_message_list(req)
    prompt_ids = _prepare_prompt(runtime.tok, messages, resolved, runtime.max_seq_len)
    prompt_tokens = jnp.array(prompt_ids)
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
        key=_make_key(req.seed),
    ):
        token_list = step.tokens.tolist()
        yield StreamFrame(
            step=step.step,
            total=step.total_steps,
            tokens=token_list,
            text=_render_with_blocks(runtime.tok, token_list),
            mask_positions=_mask_positions(step.tokens, mask_id),
        )


__all__ = ["generate_blocking", "generate_stream"]
