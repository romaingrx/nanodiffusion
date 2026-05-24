# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax>=0.6",
#     "equinox>=0.11",
#     "orbax-checkpoint>=0.11,<0.12",
#     "tensorstore>=0.1.65,<0.2",
#     "etils[epath-gcs]",
# ]
# ///
"""Benchmark Orbax + TensorStore save/restore against the production payload.

Times one save + wait_until_finished and one restore against the given URI
(local path or ``gs://...``), printing wall-time and effective MB/s.

Example:
    uv run examples/bench_checkpoint.py --uri /tmp/orbax-bench --gb 0.1
    uv run examples/bench_checkpoint.py --uri gs://my-bucket/bench --gb 6.8
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from etils import epath


def _make_state(gb: float) -> dict[str, jax.Array]:
    n_elems = int(gb * 1024**3 // 4)
    side = max(1, math.isqrt(n_elems))
    return {"big": jnp.ones((side, side), dtype=jnp.float32)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uri",
        required=True,
        help="Target directory (local path or gs:// URI).",
    )
    parser.add_argument(
        "--gb",
        type=float,
        default=6.8,
        help="Synthetic state size in GB (default matches production ckpt).",
    )
    args = parser.parse_args()

    if not args.uri.startswith("gs://"):
        local = Path(args.uri)
        if local.exists():
            shutil.rmtree(local)

    state = _make_state(args.gb)
    bytes_total = sum(int(a.size) * a.dtype.itemsize for a in jax.tree.leaves(state))
    bytes_mb = bytes_total / 1024**2
    print(f"state size: {bytes_mb:.1f} MB ({bytes_total} bytes)")

    mngr = ocp.CheckpointManager(
        epath.Path(args.uri),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            step_prefix="step",
            enable_async_checkpointing=True,
        ),
        item_names=("state",),
    )

    t0 = time.perf_counter()
    mngr.save(
        0,
        args=ocp.args.Composite(state=ocp.args.StandardSave(state)),
    )
    t_submit = time.perf_counter() - t0
    mngr.wait_until_finished()
    t_save = time.perf_counter() - t0
    print(
        f"save:    submit={t_submit:.2f}s total={t_save:.2f}s "
        f"throughput={bytes_mb / t_save:.1f} MB/s",
    )

    abstract = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), state)
    t0 = time.perf_counter()
    restored = mngr.restore(
        0,
        args=ocp.args.Composite(state=ocp.args.StandardRestore(abstract)),
    )
    t_load = time.perf_counter() - t0
    print(f"restore: total={t_load:.2f}s throughput={bytes_mb / t_load:.1f} MB/s")
    restored_bytes = sum(
        int(a.size) * a.dtype.itemsize for a in jax.tree.leaves(restored["state"])
    )
    assert restored_bytes == bytes_total, (
        f"restored payload size {restored_bytes} != saved {bytes_total}"
    )


if __name__ == "__main__":
    main()
