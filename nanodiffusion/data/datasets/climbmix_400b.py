"""ClimbMix-400B pretraining dataset."""

from nanodiffusion.data.datasets._base import register_hf_parquet

register_hf_parquet(
    name="climbmix-400b",
    repo_id="karpathy/climbmix-400b-shuffle",
    filename_pattern="shard_{index:05d}.parquet",
    num_shards=6542,
    doc="ClimbMix-400B (Karpathy). nanochat default. 6543 shards, last is val.",
)
