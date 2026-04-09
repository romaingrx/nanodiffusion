"""FineWeb-Edu sample-10BT pretraining dataset."""

from nanodiffusion.data.datasets._base import register_hf_parquet

register_hf_parquet(
    name="fineweb-edu-10bt",
    repo_id="HuggingFaceFW/fineweb-edu",
    filename_pattern="sample/10BT/{index:03d}_00000.parquet",
    num_shards=13,
    doc="FineWeb-Edu sample-10BT subset (HuggingFaceFW). 14 shards, last is val.",
)
