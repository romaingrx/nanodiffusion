"""Load a checkpoint into a frozen, inference-ready :class:`Runtime`.

Shared by the ``sample`` and ``serve`` CLI commands plus their tests.
Sits at the top level so it never has to depend on FastAPI or other
serve-only modules.
"""

import dataclasses
from collections.abc import Iterable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from nanodiffusion import sampler
from nanodiffusion.checkpoint import load_meta, load_model
from nanodiffusion.config import Config, SampleConfig
from nanodiffusion.constants import CONFIG_SIDECAR_FILENAME
from nanodiffusion.model import Transformer, transformer_skeleton
from nanodiffusion.schedule import LogLinearSchedule, NoiseSchedule
from nanodiffusion.tokenizer import Tokenizer


@dataclasses.dataclass(frozen=True)
class SampleConfigOverride:
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
    defaults: SampleConfig
    train_step: int
    max_seq_len: int
    checkpoint_path: Path


def load_runtime(
    checkpoint: Path,
    *,
    overrides: SampleConfigOverride | None = None,
) -> Runtime:
    """Read config + EMA weights + meta from a run directory.

    ``checkpoint`` is a *run* directory (the one that holds
    ``config.yaml`` at its root and Orbax-managed ``step_N/`` children);
    the Orbax manager opens it and serves the latest finalised step.
    """
    overrides = overrides if overrides is not None else SampleConfigOverride()
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

    meta = load_meta(checkpoint)
    defaults = config.sample.with_overrides(overrides)

    return Runtime(
        model=model,
        tok=tok,
        schedule=schedule,
        defaults=defaults,
        train_step=meta.step,
        max_seq_len=config.model.max_seq_len,
        checkpoint_path=checkpoint,
    )


def warmup(runtime: Runtime, *, max_lengths: Iterable[int] | None = None) -> None:
    """Drive the sampling pipeline once per ``max_length`` to warm the JIT cache.

    ``_forward`` is JIT-keyed on ``(seq_len, dtype)``, so one trace per
    distinct ``max_length`` covers every request at that length.
    Defaults to warming both the configured default and the model's
    full ``max_seq_len`` so the two most common request shapes skip
    the 5-15s compile.
    """
    lengths = (
        list(max_lengths)
        if max_lengths is not None
        else [runtime.defaults.max_length, runtime.max_seq_len]
    )
    # Prompt content is immaterial for warmup: the sampler pads out to
    # ``max_length`` with masks before calling ``_forward``, whose JIT
    # cache keys on ``(seq_len, dtype)`` — any 1-token int32 prompt traces
    # the same shape real requests will use.
    prompt = jnp.zeros(1, dtype=jnp.int32)
    for length in dict.fromkeys(lengths):  # dedupe, preserve order
        tokens = sampler.sample_tokens(
            runtime.model,
            prompt,
            schedule=runtime.schedule,
            mask_token_id=runtime.tok.mask_token_id,
            max_length=length,
            steps=1,
            key=jax.random.PRNGKey(0),
        )
        tokens.block_until_ready()
