from pathlib import Path
from typing import assert_type

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanodiffusion.chat import Conversation
from nanodiffusion.checkpoint import save_checkpoint
from nanodiffusion.config import (
    Config,
    DataConfig,
    ModelConfig,
    SFTConfig,
    SFTDatasetConfig,
)
from nanodiffusion.constants import CONFIG_SIDECAR_FILENAME, LATEST_LINK_NAME
from nanodiffusion.data.chat_datasets import CHAT_DATASETS, register_chat
from nanodiffusion.data.chat_source import ChatSource, InMemoryChatSource
from nanodiffusion.data.datasets import DownloadOptions
from nanodiffusion.data.sft_loader import SFTJaxBatch
from nanodiffusion.model import Transformer
from nanodiffusion.optimizer import make_optimizer
from nanodiffusion.schedule import LogLinearSchedule
from nanodiffusion.sft import SFTTrainStepFn, make_sft_train_step, sft_finetune
from tests._helpers import clone_state, inexact_leaves

_DUMMY_DATASETS = [SFTDatasetConfig(name="_placeholder")]


def _make_supervised_batch(seq_len: int, batch: int = 4) -> SFTJaxBatch:
    """Build a synthetic SFT batch with a prompt half and a response half."""
    tokens = jnp.tile(
        jnp.arange(seq_len, dtype=jnp.int32) % 16,
        (batch, 1),
    )
    loss_mask = jnp.tile(
        jnp.arange(seq_len) >= (seq_len // 2),
        (batch, 1),
    )
    return SFTJaxBatch(tokens=tokens, loss_mask=loss_mask)


def test_sft_train_step_decreases_loss_on_fixed_batch(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """A few aggressive-LR SFT steps on the same batch must lower the loss."""
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)
    ema_model = clone_state(model)

    sft_cfg = SFTConfig(
        datasets=_DUMMY_DATASETS,
        learning_rate=3e-3,
        warmup_steps=5,
        max_steps=100,
        weight_decay=0.0,
        grad_clip=1.0,
        ema_decay=0.99,
    )
    optimizer, _ = make_optimizer(sft_cfg)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    train_step = make_sft_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=small_config.vocab_size - 1,
        ema_decay=sft_cfg.ema_decay,
    )

    batch = _make_supervised_batch(small_config.max_seq_len, batch=4)

    losses: list[float] = []
    for _ in range(50):
        key, step_key = jax.random.split(key)
        model, ema_model, opt_state, metrics, _ = train_step(
            model, ema_model, opt_state, batch, step_key
        )
        losses.append(float(metrics.loss))

    early = float(np.mean(losses[:5]))
    late = float(np.mean(losses[-5:]))
    assert late < early, f"loss did not decrease: {early:.3f} -> {late:.3f}"


def test_sft_train_step_is_deterministic(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """Same key + same batch + same init → bitwise identical updates."""
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)

    optimizer, _ = make_optimizer(
        SFTConfig(datasets=_DUMMY_DATASETS, warmup_steps=2, max_steps=10)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_sft_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=small_config.vocab_size - 1,
        ema_decay=0.9,
    )

    batch = _make_supervised_batch(small_config.max_seq_len, batch=2)

    # Fresh PRNGKey per call — ``donate="all"`` on the JIT'd train step
    # consumes the key buffer, so sharing one across two calls would
    # hit "buffer has been deleted or donated".
    m1, e1, _o1, mx1, _ = train_step(
        clone_state(model),
        clone_state(model),
        clone_state(opt_state),
        batch,
        jax.random.PRNGKey(123),
    )
    m2, e2, _o2, mx2, _ = train_step(
        clone_state(model),
        clone_state(model),
        clone_state(opt_state),
        batch,
        jax.random.PRNGKey(123),
    )

    assert float(mx1.loss) == float(mx2.loss)
    for a, b in zip(inexact_leaves(m1), inexact_leaves(m2), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(inexact_leaves(e1), inexact_leaves(e2), strict=True):
        np.testing.assert_array_equal(a, b)


def test_sft_train_step_prompt_positions_have_zero_embedding_grad(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """Full-stack regression: with loss_mask=all-False, gradients w.r.t. the
    entire model (post-JIT) are zero. Mirrors Slice 1's load-bearing assertion
    but through the full ``train_step`` path so any JIT reordering that would
    accidentally leak gradient from prompt positions is caught.
    """
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)

    optimizer, _ = make_optimizer(
        SFTConfig(datasets=_DUMMY_DATASETS, warmup_steps=1, max_steps=10)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_sft_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=small_config.vocab_size - 1,
        ema_decay=0.9,
    )

    tokens = jnp.tile(
        jnp.arange(small_config.max_seq_len, dtype=jnp.int32)
        % (small_config.vocab_size - 1),
        (2, 1),
    )
    loss_mask = jnp.zeros((2, small_config.max_seq_len), dtype=jnp.bool_)
    batch = SFTJaxBatch(tokens=tokens, loss_mask=loss_mask)
    step_key = jax.random.PRNGKey(5)

    # Clone the donated inputs so the original ``model`` stays live
    # for the post-step comparisons below.
    new_model, _new_ema, _opt, metrics, _ = train_step(
        clone_state(model), clone_state(model), opt_state, batch, step_key
    )
    assert float(metrics.loss) == 0.0
    # With zero loss, the optimizer should have made no change to the model.
    for before, after in zip(
        inexact_leaves(model), inexact_leaves(new_model), strict=True
    ):
        np.testing.assert_array_equal(before, after)


def test_make_sft_train_step_narrows_via_sfttrainstepfn_annotation(
    small_config: ModelConfig, key: jax.Array
) -> None:
    """Generic narrowing: a ``SFTTrainStepFn[Transformer]`` target annotation
    pins ``M = Transformer`` so each return position narrows at call sites.
    """
    key, model_key = jax.random.split(key)
    model = Transformer(small_config, key=model_key)

    optimizer, _ = make_optimizer(
        SFTConfig(datasets=_DUMMY_DATASETS, warmup_steps=2, max_steps=10, ema_decay=0.9)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    train_step: SFTTrainStepFn[Transformer] = make_sft_train_step(
        optimizer,
        schedule=LogLinearSchedule(),
        mask_token_id=small_config.vocab_size - 1,
        ema_decay=0.9,
    )

    batch = _make_supervised_batch(small_config.max_seq_len, batch=2)
    key, step_key = jax.random.split(key)
    new_model, new_ema, _new_opt_state, metrics, _ = train_step(
        model, clone_state(model), opt_state, batch, step_key
    )

    assert_type(new_model, Transformer)
    assert_type(new_ema, Transformer)
    assert type(new_model) is Transformer
    assert type(new_ema) is Transformer
    assert jnp.isfinite(metrics.loss)


def _make_sft_conv(i: int) -> Conversation:
    return {
        "messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ],
    }


def _tiny_inmem_factory(
    data_dir: Path,
    *,
    download: bool = True,
    download_options: DownloadOptions | None = None,
) -> ChatSource:
    del data_dir, download, download_options
    return InMemoryChatSource([_make_sft_conv(i) for i in range(8)])


@pytest.fixture
def tiny_chat_factory() -> str:
    """Register a tiny in-memory chat dataset and return its name.

    Adds a one-off entry to ``CHAT_DATASETS`` so ``sft_finetune`` can
    resolve a dataset via the registry without touching the network.
    Not automatically unregistered — the registry is a module-level
    dict, and the name is unique so re-registration is a no-op.
    """
    name = "tiny_inmem"
    if name not in CHAT_DATASETS:
        register_chat(name)(_tiny_inmem_factory)
    return name


def _tiny_config_full_vocab(small_config: ModelConfig) -> ModelConfig:
    """Widen ``small_config``'s vocab to match the real GPT-2 tokenizer.

    The general-purpose ``small_config`` fixture caps vocab at 256 for
    unit tests that feed synthetic token IDs directly; the SFT
    end-to-end tests pipe through :class:`Tokenizer` whose token IDs
    reach 50263, so a mismatched-width embedding table would gather
    out-of-bounds, which under the new NaN loss guard fails the run
    loudly instead of silently training on garbage.
    """
    return small_config.model_copy(update={"vocab_size": 50264})


def test_sft_finetune_end_to_end_smoke(
    small_config: ModelConfig,
    tmp_path: Path,
    tiny_chat_factory: str,
) -> None:
    """A fresh pretrain-shaped checkpoint → sft_finetune → run dir artifacts."""
    sft_model_config = _tiny_config_full_vocab(small_config)
    key = jax.random.PRNGKey(0)
    model_key, _ = jax.random.split(key)
    model = Transformer(sft_model_config, key=model_key)
    optimizer, _ = make_optimizer(
        SFTConfig(datasets=_DUMMY_DATASETS, warmup_steps=1, max_steps=5)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    ckpt_dir = tmp_path / "pretrain_latest"
    save_checkpoint(
        ckpt_dir,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=jax.random.PRNGKey(0),
        step=0,
        cursor=None,
    )
    # Drop a config.yaml alongside so _resolve_model_config reads it.
    import yaml  # noqa: PLC0415

    ckpt_config = Config(model=sft_model_config)
    (ckpt_dir / CONFIG_SIDECAR_FILENAME).write_text(
        yaml.dump(ckpt_config.model_dump(mode="json"))
    )

    sft_config = Config(
        model=sft_model_config,
        data=DataConfig(dataset="_unused", data_dir=tmp_path),
        sft=SFTConfig(
            warmup_steps=1,
            max_steps=3,
            batch_size=2,
            log_every=1,
            save_every=2,
            run_dir=tmp_path / "sft_runs",
            datasets=[SFTDatasetConfig(name=tiny_chat_factory)],
            prefetch_size=1,
        ),
    )

    run_dir = sft_finetune(sft_config, pretrain_checkpoint=ckpt_dir)
    assert run_dir.exists()
    assert (run_dir / CONFIG_SIDECAR_FILENAME).exists()
    assert (run_dir / LATEST_LINK_NAME).exists()
    # max_steps=3 with save_every=2 → step_2/ exists, step_3/ final save
    step_dirs = sorted(d.name for d in run_dir.iterdir() if d.name.startswith("step_"))
    assert step_dirs, f"no step_ dirs in {run_dir}"


def test_sft_finetune_requires_exactly_one_start_point(
    small_config: ModelConfig, tmp_path: Path, tiny_chat_factory: str
) -> None:
    """Passing both or neither of the start-point kwargs must fail fast."""
    sft_config = Config(
        model=small_config,
        sft=SFTConfig(
            warmup_steps=1,
            max_steps=3,
            batch_size=2,
            datasets=[SFTDatasetConfig(name=tiny_chat_factory)],
            run_dir=tmp_path / "sft_runs",
        ),
    )
    with pytest.raises(ValueError, match="got neither"):
        sft_finetune(sft_config)
    with pytest.raises(ValueError, match="got both"):
        sft_finetune(
            sft_config,
            pretrain_checkpoint=tmp_path / "a",
            resume_from=tmp_path / "b",
        )


def test_sft_finetune_resumes_from_saved_sft_checkpoint(
    small_config: ModelConfig, tmp_path: Path, tiny_chat_factory: str
) -> None:
    """Run → interrupt → resume: the second run must pick up where the first left off.

    End-to-end guarantee that opt_state, EMA, step counter, and SFT
    cursor all survive the save/load roundtrip. The saved checkpoint's
    cursor drives the loader's permutation fast-forward so resume does
    not re-ingest the same conversations.
    """
    import yaml  # noqa: PLC0415

    sft_model_config = _tiny_config_full_vocab(small_config)
    pretrain_dir = tmp_path / "pretrain_latest"
    key = jax.random.PRNGKey(0)
    model = Transformer(sft_model_config, key=key)
    pretrain_optimizer, _ = make_optimizer(
        SFTConfig(datasets=_DUMMY_DATASETS, warmup_steps=1, max_steps=5)
    )
    pretrain_opt_state = pretrain_optimizer.init(
        eqx.filter(model, eqx.is_inexact_array)
    )
    save_checkpoint(
        pretrain_dir,
        model=model,
        ema_model=model,
        opt_state=pretrain_opt_state,
        key=jax.random.PRNGKey(0),
        step=0,
        cursor=None,
    )
    (pretrain_dir / CONFIG_SIDECAR_FILENAME).write_text(
        yaml.dump(Config(model=sft_model_config).model_dump(mode="json"))
    )

    # First run stops cleanly at step 2 (max_steps == save_every), so
    # the periodic save and the end-of-training save collapse into a
    # single step_2 checkpoint and there's no step_4/step_6 lying
    # around to collide with the resumed run's new saves.
    first_config = Config(
        model=sft_model_config,
        data=DataConfig(dataset="_unused", data_dir=tmp_path),
        sft=SFTConfig(
            warmup_steps=1,
            max_steps=2,
            batch_size=2,
            log_every=1,
            save_every=2,
            run_dir=tmp_path / "sft_runs",
            datasets=[SFTDatasetConfig(name=tiny_chat_factory)],
            prefetch_size=1,
        ),
    )
    first_run = sft_finetune(first_config, pretrain_checkpoint=pretrain_dir)
    mid_ckpt = first_run / "step_2"
    assert mid_ckpt.exists(), sorted(p.name for p in first_run.iterdir())

    # Second run resumes and trains to step 6 → expect fresh step_4 + step_6.
    resumed_config = first_config.model_copy(
        update={"sft": first_config.sft.model_copy(update={"max_steps": 6})}
    )
    second_run = sft_finetune(resumed_config, resume_from=mid_ckpt)
    # Resume reuses the same run dir per resolve_run_dir semantics.
    assert second_run == first_run

    final_ckpts = sorted(
        d.name for d in second_run.iterdir() if d.name.startswith("step_")
    )
    assert "step_2" in final_ckpts
    assert "step_4" in final_ckpts
    assert "step_6" in final_ckpts
