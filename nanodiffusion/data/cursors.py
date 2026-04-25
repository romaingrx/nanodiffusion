"""Resume cursors for pretrain and SFT data loaders.

A cursor is a frozen, JSON-serialisable snapshot of where a loader left
off so that a checkpoint can fast-forward past the already-consumed part
of its stream on resume. The two paradigms consume different kinds of
streams (sharded parquet row groups vs a shuffled in-memory permutation),
so they carry different fields. A ``kind`` discriminator lets pydantic
round-trip either variant through :class:`CheckpointMeta` without an
opaque ``dict[str, int]`` escape hatch.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class PretrainCursor(BaseModel):
    """Exact next-token cursor for the sharded parquet pretrain loader.

    ``shard_idx`` indexes into the loader's shard list and
    ``row_group_idx`` is the parquet row group that contains the next
    token to emit. ``doc_idx`` indexes into that row group's non-null
    text rows and ``token_offset`` indexes into the tokenized document
    including its synthetic trailing EOS separator. Cursors are allowed
    to point just past the end of a row group; the source canonicalizes
    that to the next row group when resuming.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["pretrain"] = "pretrain"
    epoch: int = Field(ge=1)
    shard_idx: int = Field(ge=0)
    row_group_idx: int = Field(ge=0)
    doc_idx: int = Field(ge=0)
    token_offset: int = Field(ge=0)

    def cursor_for_batch_doc(self, local_doc_idx: int) -> "PretrainCursor":
        """Return a document-start cursor for a doc inside this source batch."""
        if local_doc_idx < 0:
            msg = f"local_doc_idx must be >= 0, got {local_doc_idx}"
            raise ValueError(msg)
        return self.model_copy(
            update={
                "doc_idx": self.doc_idx + local_doc_idx,
                "token_offset": 0,
            }
        )

    def same_document(self, other: "PretrainCursor") -> bool:
        """Compare document identity while ignoring token offset."""
        return (
            self.epoch == other.epoch
            and self.shard_idx == other.shard_idx
            and self.row_group_idx == other.row_group_idx
            and self.doc_idx == other.doc_idx
        )

    def token_offset_for(self, document: "PretrainCursor") -> int:
        """Return resume token offset if ``document`` is this cursor's doc."""
        return self.token_offset if self.same_document(document) else 0

    def with_token_offset(self, token_offset: int) -> "PretrainCursor":
        """Return this document cursor at ``token_offset``."""
        if token_offset < 0:
            msg = f"token_offset must be >= 0, got {token_offset}"
            raise ValueError(msg)
        return self.model_copy(update={"token_offset": token_offset})


class SFTCursor(BaseModel):
    """Exact next-row cursor for the shuffled SFT loader.

    ``permutation_idx`` is the next position inside the current
    epoch's seeded permutation. It may equal ``len(source)``, in which
    case resume rolls cleanly into the next epoch before yielding.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["sft"] = "sft"
    epoch: int = Field(ge=1)
    permutation_idx: int = Field(ge=0)


type LoaderCursor = Annotated[
    PretrainCursor | SFTCursor,
    Field(discriminator="kind"),
]
