"""Load a checkpoint into a frozen, inference-ready :class:`Runtime`.

Shared by the CLI ``serve`` command and its tests; any future non-CLI
consumer (notebook, benchmark script) can reuse this without touching
FastAPI. Intentionally does not depend on ``nanodiffusion.serve.app`` so
generation tests can exercise the full model path without FastAPI.
"""

import dataclasses
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from nanodiffusion import sampler
from nanodiffusion.checkpoint import CheckpointMeta, load_model
from nanodiffusion.config import Config, SampleConfig
from nanodiffusion.constants import CONFIG_SIDECAR_FILENAME, META_FILENAME
from nanodiffusion.model import Transformer, transformer_skeleton
from nanodiffusion.schedule import LogLinearSchedule, NoiseSchedule
from nanodiffusion.serve.protocol import SampleDefaults
from nanodiffusion.tokenizer import Tokenizer


@dataclasses.dataclass(frozen=True)
class SampleDefaultsOverride:
    """Optional per-field overrides layered on top of ``config.sample``."""

    steps: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    max_length: int | None = None


@dataclasses.dataclass(frozen=True)
class Runtime:
    model: Transformer
    tok: Tokenizer
    schedule: NoiseSchedule
    defaults: SampleDefaults
    train_step: int
    max_seq_len: int
    checkpoint_path: Path


def _resolve_defaults(
    base: SampleConfig, overrides: SampleDefaultsOverride
) -> SampleDefaults:
    return SampleDefaults(
        steps=overrides.steps if overrides.steps is not None else base.steps,
        temperature=overrides.temperature
        if overrides.temperature is not None
        else base.temperature,
        top_k=overrides.top_k if overrides.top_k is not None else base.top_k,
        top_p=overrides.top_p if overrides.top_p is not None else base.top_p,
        max_length=overrides.max_length
        if overrides.max_length is not None
        else base.max_length,
    )


def load_runtime(
    checkpoint: Path,
    *,
    overrides: SampleDefaultsOverride | None = None,
) -> Runtime:
    """Read config + EMA weights + meta from ``checkpoint`` into a :class:`Runtime`."""
    overrides = overrides if overrides is not None else SampleDefaultsOverride()
    sidecar = checkpoint / CONFIG_SIDECAR_FILENAME
    if not sidecar.exists():
        msg = f"checkpoint missing {CONFIG_SIDECAR_FILENAME}: {checkpoint}"
        raise FileNotFoundError(msg)

    config = Config.from_yaml(sidecar)
    tok = Tokenizer()
    schedule = LogLinearSchedule()

    skeleton = transformer_skeleton(config.model)
    model = load_model(checkpoint, model_skeleton=skeleton, which="ema")
    model = eqx.nn.inference_mode(model, value=True)

    meta = CheckpointMeta.model_validate_json((checkpoint / META_FILENAME).read_text())
    defaults = _resolve_defaults(config.sample, overrides)

    return Runtime(
        model=model,
        tok=tok,
        schedule=schedule,
        defaults=defaults,
        train_step=meta.step,
        max_seq_len=config.model.max_seq_len,
        checkpoint_path=checkpoint,
    )


def warmup(runtime: Runtime) -> None:
    """Drive the full sampling pipeline once so the JIT cache is hot.

    Uses ``steps=1`` through the real ``sample_tokens`` entry point so
    every JIT-compiled region (including the sampler's module-scope
    ``_forward``) is cached against the serving shape. First real
    request at the same ``max_length`` skips the 5-15s compile;
    requests with a different ``max_length`` still trigger a fresh
    compile.
    """
    prompt = jnp.zeros(1, dtype=jnp.int32)
    tokens = sampler.sample_tokens(
        runtime.model,
        prompt,
        schedule=runtime.schedule,
        mask_token_id=runtime.tok.mask_token_id,
        max_length=runtime.defaults.max_length,
        steps=1,
        key=jax.random.PRNGKey(0),
    )
    tokens.block_until_ready()
