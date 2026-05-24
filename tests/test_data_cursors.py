import pytest

from nanodiffusion.data.cursors import PretrainCursor


def test_pretrain_cursor_resolves_batch_doc_and_resume_offset() -> None:
    batch_position = PretrainCursor(
        epoch=2,
        shard_idx=3,
        row_group_idx=5,
        doc_idx=10,
        token_offset=0,
    )
    resume = PretrainCursor(
        epoch=2,
        shard_idx=3,
        row_group_idx=5,
        doc_idx=12,
        token_offset=7,
    )

    first_doc = batch_position.cursor_for_batch_doc(0)
    resumed_doc = batch_position.cursor_for_batch_doc(2)

    assert first_doc == batch_position
    assert resumed_doc.doc_idx == 12
    assert resume.token_offset_for(first_doc) == 0
    assert resume.token_offset_for(resumed_doc) == 7
    assert resumed_doc.with_token_offset(7) == resume


def test_pretrain_cursor_rejects_negative_offsets() -> None:
    cursor = PretrainCursor(
        epoch=1,
        shard_idx=0,
        row_group_idx=0,
        doc_idx=0,
        token_offset=0,
    )

    with pytest.raises(ValueError, match="local_doc_idx"):
        cursor.cursor_for_batch_doc(-1)
    with pytest.raises(ValueError, match="token_offset"):
        cursor.with_token_offset(-1)
