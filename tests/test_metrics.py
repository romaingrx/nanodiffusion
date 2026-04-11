import jax.numpy as jnp
import pytest

from nanodiffusion.metrics import (
    CoreHostMetrics,
    CoreStepMetrics,
    ReportMetrics,
    SFTHostExtras,
)


def test_core_host_metrics_omits_none_optionals() -> None:
    metrics = CoreHostMetrics(
        loss=1.0,
        grad_norm=2.0,
        param_norm=3.0,
        grad_finite=1.0,
        lr=1e-3,
        tok_per_s=42,
        num_devices=8,
    )

    assert metrics.to_dict() == {
        "loss": 1.0,
        "grad_norm": 2.0,
        "param_norm": 3.0,
        "grad_finite": 1.0,
        "lr": 1e-3,
        "tok_per_s": 42,
        "num_devices": 8,
    }


def test_report_metrics_merges_sft_extras() -> None:
    report = ReportMetrics(
        core=CoreHostMetrics(
            loss=1.0,
            grad_norm=2.0,
            param_norm=3.0,
            grad_finite=1.0,
            lr=1e-3,
            tok_per_s=42,
            num_devices=8,
            hbm_used_gb=10.5,
        ),
        extras=SFTHostExtras(supervised_tok_per_s=21),
    )

    payload = report.to_dict()
    assert payload["supervised_tok_per_s"] == 21
    assert payload["hbm_used_gb"] == 10.5


def test_report_metrics_pretrain_extras_are_empty() -> None:
    report = ReportMetrics(
        core=CoreHostMetrics(
            loss=1.0,
            grad_norm=2.0,
            param_norm=3.0,
            grad_finite=1.0,
            lr=1e-3,
            tok_per_s=42,
            num_devices=1,
        )
    )

    assert "supervised_tok_per_s" not in report.to_dict()


def test_core_host_metrics_from_step_metrics_collects_runtime_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDevice:
        def memory_stats(self) -> dict[str, float]:
            return {
                "bytes_in_use": 12.34e9,
                "peak_bytes_in_use": 23.45e9,
            }

    monkeypatch.setattr("nanodiffusion.metrics.jax.device_count", lambda: 8)
    monkeypatch.setattr("nanodiffusion.metrics.jax.devices", lambda: [_FakeDevice()])

    metrics = CoreHostMetrics.from_step_metrics(
        CoreStepMetrics(
            loss=jnp.array(1.0),
            grad_norm=jnp.array(2.0),
            param_norm=jnp.array(3.0),
            grad_finite=jnp.array(1.0),
        ),
        lr_schedule=lambda step: jnp.asarray(step / 1000),
        step=5,
        tok_per_s=42,
    )

    assert metrics.num_devices == 8
    assert metrics.lr == pytest.approx(0.005)
    assert metrics.hbm_used_gb == 12.34
    assert metrics.hbm_peak_gb == 23.45


def test_sft_host_extras_from_window() -> None:
    extras = SFTHostExtras.from_window(supervised_tokens_in_window=21, elapsed=1.0)
    assert extras.to_dict() == {"supervised_tok_per_s": 21}
