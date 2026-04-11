"""Typed metric payloads shared by training and reporting."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from nanodiffusion.types import Scalar

type MetricValue = float | int | str


class CoreStepMetrics(eqx.Module):
    """Per-step optimization metrics emitted directly by the JIT'd train step."""

    loss: Scalar
    grad_norm: Scalar
    param_norm: Scalar
    grad_finite: Scalar


@dataclasses.dataclass(frozen=True, slots=True)
class CoreHostMetrics:
    """Host-side training metrics shared by pretrain and SFT."""

    loss: float
    grad_norm: float
    param_norm: float
    grad_finite: float
    lr: float
    tok_per_s: int
    num_devices: int
    hbm_used_gb: float | None = None
    hbm_peak_gb: float | None = None

    @classmethod
    def from_step_metrics(
        cls,
        step_metrics: CoreStepMetrics,
        *,
        lr_schedule: optax.Schedule,
        step: int,
        tok_per_s: int,
    ) -> "CoreHostMetrics":
        mem = jax.devices()[0].memory_stats()
        return cls(
            loss=float(step_metrics.loss),
            grad_norm=float(step_metrics.grad_norm),
            param_norm=float(step_metrics.param_norm),
            grad_finite=float(step_metrics.grad_finite),
            lr=float(jnp.asarray(lr_schedule(step)).item()),
            tok_per_s=tok_per_s,
            num_devices=jax.device_count(),
            hbm_used_gb=None if mem is None else round(mem["bytes_in_use"] / 1e9, 2),
            hbm_peak_gb=(
                None if mem is None else round(mem["peak_bytes_in_use"] / 1e9, 2)
            ),
        )

    def to_dict(self) -> dict[str, MetricValue]:
        out: dict[str, MetricValue] = {
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "param_norm": self.param_norm,
            "grad_finite": self.grad_finite,
            "lr": self.lr,
            "tok_per_s": self.tok_per_s,
            "num_devices": self.num_devices,
        }
        if self.hbm_used_gb is not None:
            out["hbm_used_gb"] = self.hbm_used_gb
        if self.hbm_peak_gb is not None:
            out["hbm_peak_gb"] = self.hbm_peak_gb
        return out


@dataclasses.dataclass(frozen=True, slots=True)
class NoHostExtras:
    """No-op extras payload used by pretraining."""

    def to_dict(self) -> dict[str, MetricValue]:
        return {}


@dataclasses.dataclass(frozen=True, slots=True)
class SFTHostExtras:
    """Additional host-side SFT-only metrics."""

    supervised_tok_per_s: int

    @classmethod
    def from_window(
        cls, *, supervised_tokens_in_window: int, elapsed: float
    ) -> "SFTHostExtras":
        return cls(
            supervised_tok_per_s=int(supervised_tokens_in_window / max(elapsed, 1e-9))
        )

    def to_dict(self) -> dict[str, MetricValue]:
        return {"supervised_tok_per_s": self.supervised_tok_per_s}


@dataclasses.dataclass(frozen=True, slots=True)
class ReportMetrics:
    """Composable reporting payload assembled at the reporter boundary."""

    core: CoreHostMetrics
    extras: NoHostExtras | SFTHostExtras = dataclasses.field(
        default_factory=NoHostExtras
    )

    def to_dict(self) -> dict[str, MetricValue]:
        return self.core.to_dict() | self.extras.to_dict()
