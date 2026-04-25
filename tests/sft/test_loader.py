from itertools import islice
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from nanodiffusion.chat import Conversation, render_conversation
from nanodiffusion.checkpoint import (
    CheckpointMeta,
    load_checkpoint,
    save_checkpoint,
)
from nanodiffusion.config import ModelConfig, SFTConfig, SFTDatasetConfig
from nanodiffusion.data.chat_source import InMemoryChatSource
from nanodiffusion.data.cursors import SFTCursor
from nanodiffusion.data.sft_loader import (
    SFTBatchOutput,
    SFTJaxBatch,
    sft_loader,
)
from nanodiffusion.model.transformer import Transformer
from nanodiffusion.optimizer import make_optimizer
from nanodiffusion.tokenizer import Tokenizer

_DUMMY_DATASETS = [SFTDatasetConfig(name="_placeholder")]


def _short_conv(i: int) -> Conversation:
    return {
        "messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ],
    }


def _make_source(n: int = 32) -> InMemoryChatSource:
    return InMemoryChatSource([_short_conv(i) for i in range(n)])


def test_batch_shape_and_dtype(tok: Tokenizer) -> None:
    src = _make_source()
    loader = sft_loader(src, tok, batch_size=4, seq_len=32)
    batch = next(loader)
    assert batch.tokens.shape == (4, 32)
    assert batch.loss_mask.shape == (4, 32)
    assert batch.tokens.dtype == np.int32
    assert batch.loss_mask.dtype == np.bool_


def test_prompt_positions_are_unsupervised(tok: Tokenizer) -> None:
    """Every `loss_mask=1` position must correspond to an assistant token."""
    src = _make_source()
    loader = sft_loader(src, tok, batch_size=2, seq_len=64)
    batch = next(loader)
    for row in range(2):
        supervised_positions = np.flatnonzero(batch.loss_mask[row])
        assert len(supervised_positions) > 0
        # Find the assistant span bounds via the rendered reference.
        for src_idx in range(len(src)):
            ids, mask = render_conversation(tok, src[src_idx])
            if list(batch.tokens[row, : len(ids)]) == ids:
                assert list(batch.loss_mask[row, : len(ids)]) == [bool(m) for m in mask]
                break
        else:
            pytest.fail("batch row did not correspond to any source conversation")


def test_pad_tail_is_eos_and_unmasked(tok: Tokenizer) -> None:
    src = _make_source(n=4)
    loader = sft_loader(src, tok, batch_size=2, seq_len=64)
    batch = next(loader)
    eos = tok.eos_token_id
    for row in range(2):
        ids_for_row = None
        for idx in range(len(src)):
            ids, _mask = render_conversation(tok, src[idx])
            if list(batch.tokens[row, : len(ids)]) == ids:
                ids_for_row = ids
                break
        assert ids_for_row is not None
        pad_tokens = batch.tokens[row, len(ids_for_row) :]
        pad_mask = batch.loss_mask[row, len(ids_for_row) :]
        assert (pad_tokens == eos).all()
        assert not pad_mask.any()


def test_oversized_conversations_are_skipped(tok: Tokenizer) -> None:
    """A conversation longer than seq_len must never appear in any batch."""
    big_content = "word " * 200
    big_conv: Conversation = {
        "messages": [
            {"role": "user", "content": big_content},
            {"role": "assistant", "content": big_content},
        ],
    }
    small = [_short_conv(i) for i in range(6)]
    # Interleave the giant conversation so the loader has to skip it.
    src = InMemoryChatSource([*small[:3], big_conv, *small[3:]])
    loader = sft_loader(src, tok, batch_size=2, seq_len=32)
    batches = list(islice(loader, 4))

    big_ids, _ = render_conversation(tok, big_conv)
    big_head = tuple(big_ids[:8])
    for batch in batches:
        for row in range(2):
            assert tuple(batch.tokens[row, :8].tolist()) != big_head


def test_too_many_oversized_skips_raise(tok: Tokenizer) -> None:
    """A run of consecutive oversized conversations exhausts the skip budget.

    max_empty_passes is the safety net against a dataset where nothing
    fits the configured seq_len — firing a RuntimeError beats silently
    spinning forever.
    """
    big_content = "word " * 200
    big: Conversation = {
        "messages": [
            {"role": "user", "content": big_content},
            {"role": "assistant", "content": big_content},
        ],
    }
    src = InMemoryChatSource([big] * 10)
    loader = sft_loader(src, tok, batch_size=2, seq_len=16, max_empty_passes=5)
    with pytest.raises(RuntimeError, match="skipped"):
        next(loader)


def test_empty_assistant_content_still_trains_on_end_delimiter(
    tok: Tokenizer,
) -> None:
    """Empty assistant string is still supervised via ``<|assistant_end|>``.

    render_conversation marks the closing delimiter as supervised, so an
    assistant turn with empty content produces a mask with at least one
    '1'. That's intentional — the model learns to emit the stop token —
    and means the loader does not need a separate 'no supervision'
    skip path beyond the already-validated role alternation.
    """
    empty_asst: Conversation = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ],
    }
    _ids, mask = render_conversation(tok, empty_asst)
    assert sum(mask) >= 1


def test_determinism_same_seed(tok: Tokenizer) -> None:
    src = _make_source()
    it1 = sft_loader(src, tok, batch_size=3, seq_len=48, seed=7)
    it2 = sft_loader(src, tok, batch_size=3, seq_len=48, seed=7)
    a = list(islice(it1, 4))
    b = list(islice(it2, 4))
    for ba, bb in zip(a, b, strict=True):
        np.testing.assert_array_equal(ba.tokens, bb.tokens)
        np.testing.assert_array_equal(ba.loss_mask, bb.loss_mask)
        assert ba.state == bb.state


def test_different_seeds_diverge(tok: Tokenizer) -> None:
    src = _make_source()
    a = next(sft_loader(src, tok, batch_size=3, seq_len=48, seed=1))
    b = next(sft_loader(src, tok, batch_size=3, seq_len=48, seed=2))
    assert not np.array_equal(a.tokens, b.tokens)


def test_epoch_wraps_around(tok: Tokenizer) -> None:
    """Iterating past len(source) increments the epoch counter."""
    src = _make_source(n=5)
    loader = sft_loader(src, tok, batch_size=2, seq_len=48, seed=3)
    batches = list(islice(loader, 4))  # 4 batches of 2 rows each = 8 rows > 5 src
    final_epoch = batches[-1].state.epoch
    assert final_epoch >= 2


def test_to_jax_drops_cursor(tok: Tokenizer) -> None:
    """SFTJaxBatch must be a clean pytree without the host-side cursor."""
    src = _make_source()
    loader = sft_loader(src, tok, batch_size=2, seq_len=32)
    batch = next(loader)
    jax_batch = batch.to_jax()
    assert isinstance(jax_batch, SFTJaxBatch)
    assert jax_batch.tokens.shape == (2, 32)
    assert jax_batch.loss_mask.shape == (2, 32)
    assert jax_batch.loss_mask.dtype == jnp.bool_


def test_cursor_roundtrip_through_checkpoint(
    tok: Tokenizer, small_config: ModelConfig, tmp_path: Path
) -> None:
    """Save the loader's cursor via CheckpointMeta, reload, continue.

    Verifies the :class:`SFTCursor` discriminated-union variant survives
    the pydantic JSON roundtrip, and that resuming from the cursor
    yields the same next batch as continuing a non-saved loader.
    """
    src = _make_source()
    first = next(sft_loader(src, tok, batch_size=2, seq_len=32, seed=9))
    cursor = first.state
    assert isinstance(cursor, SFTCursor)
    assert cursor.permutation_idx == 2

    import jax  # noqa: PLC0415

    model_key = jax.random.PRNGKey(0)
    model = Transformer(small_config, key=model_key)
    optimizer, _ = make_optimizer(
        SFTConfig(datasets=_DUMMY_DATASETS, warmup_steps=1, max_steps=10)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    ckpt_dir = tmp_path / "step_1"
    save_checkpoint(
        ckpt_dir,
        model=model,
        ema_model=model,
        opt_state=opt_state,
        key=jax.random.PRNGKey(0),
        step=1,
        cursor=cursor,
    )
    _m, _e, _o, _k, meta = load_checkpoint(
        ckpt_dir,
        model_skeleton=model,
        opt_state_builder=lambda m: optimizer.init(eqx.filter(m, eqx.is_inexact_array)),
    )
    assert isinstance(meta, CheckpointMeta)
    assert meta.cursor == cursor
    assert isinstance(meta.cursor, SFTCursor)

    resumed = next(
        sft_loader(src, tok, batch_size=2, seq_len=32, seed=9, resume_state=meta.cursor)
    )
    # The resumed iterator must produce the *next* batch the unrolled iterator
    # would have produced — same tokens, same mask, advanced cursor.
    fresh = sft_loader(src, tok, batch_size=2, seq_len=32, seed=9)
    next(fresh)
    expected_second = next(fresh)
    np.testing.assert_array_equal(resumed.tokens, expected_second.tokens)
    np.testing.assert_array_equal(resumed.loss_mask, expected_second.loss_mask)


def test_rejects_non_positive_batch_size(tok: Tokenizer) -> None:
    src = _make_source()
    with pytest.raises(ValueError, match="batch_size"):
        next(sft_loader(src, tok, batch_size=0, seq_len=32))


def test_batch_output_is_frozen(tok: Tokenizer) -> None:
    src = _make_source()
    batch = next(sft_loader(src, tok, batch_size=2, seq_len=32))
    assert isinstance(batch, SFTBatchOutput)
    with pytest.raises(Exception):  # noqa: B017, PT011 - frozen dataclass error
        batch.tokens = np.zeros_like(batch.tokens)  # type: ignore[misc]


def test_resume_state_is_sft_cursor(tok: Tokenizer) -> None:
    """The cursor produced must be an :class:`SFTCursor` with the right shape."""
    src = _make_source()
    loader = sft_loader(src, tok, batch_size=2, seq_len=32)
    batch = next(loader)
    assert isinstance(batch.state, SFTCursor)
    assert batch.state.kind == "sft"
    assert batch.state.epoch >= 1
    assert batch.state.permutation_idx >= 0
