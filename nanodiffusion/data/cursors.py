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
    """Cursor for the sharded parquet pretrain loader.

    ``shard_idx`` indexes into the loader's shard list and
    ``row_group_idx`` is the parquet row group already consumed. The
    loader resumes at the *next* row group, so saving on the last row
    group of an epoch rolls cleanly into ``epoch + 1``.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["pretrain"] = "pretrain"
    epoch: int = Field(ge=1)
    shard_idx: int = Field(ge=0)
    row_group_idx: int = Field(ge=0)


class SFTCursor(BaseModel):
    """Cursor for the shuffled SFT loader.

    ``permutation_idx`` is the position already consumed inside the
    current epoch's seeded permutation; the loader resumes at
    ``permutation_idx + 1`` and rolls into the next epoch if that
    position is past the end.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["sft"] = "sft"
    epoch: int = Field(ge=1)
    permutation_idx: int = Field(ge=0)


type LoaderCursor = Annotated[
    PretrainCursor | SFTCursor,
    Field(discriminator="kind"),
]
