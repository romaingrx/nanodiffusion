#!/usr/bin/env bash
# Launch pretrain on this TPU VM under tmux (survives ssh disconnect).
#
# Usage (fresh run, first time or after VM recreate):
#   bash launch.sh
#
# Override the GCS bucket (must be same region as TPU for fast writes):
#   GCS_BUCKET=nanodiffusion-runs-other bash launch.sh
#
# Usage (resume after preemption / manual kill):
#   bash launch.sh --resume
#
# Wandb auth: ~/.netrc must contain api.wandb.ai credentials (run
# `wandb login` once interactively or write the netrc out of band).
#
# Re-invoking while a run is active replaces it; the prior tmux session is
# killed first so this is safe to call repeatedly.

set -euo pipefail

RESUME="${1:-}"
CONFIG="${CONFIG:-configs/medium.yaml}"
SESSION="${SESSION:-nano}"
WANDB_PROJECT="${WANDB_PROJECT:-nanodiffusion}"
WANDB_ENTITY="${WANDB_ENTITY:-romaingrx}"
ENV_FILE="${ENV_FILE:-$HOME/.nanodiffusion-env}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$HOME/nanodiffusion}"

# Bootstrap: repo sync, deps, GCS mount, env file. Idempotent.
bash "$SCRIPT_DIR/prepare_tpu.sh" "$RESUME"

# On --resume, point at the newest run dir. Orbax's CheckpointManager
# picks the latest finalised step itself, so no step-N discovery here.
RESUME_FLAG=""
if [ "$RESUME" = "--resume" ]; then
    LATEST_RUN=$(find "$REPO_DIR/runs/pretrain" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
        | sort | tail -1)
    if [ -n "$LATEST_RUN" ]; then
        RESUME_FLAG="--resume-from $LATEST_RUN"
    fi
fi

# Replace any prior run. Safe because nothing outside this session is writing
# to the same run_dir.
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
fi

# tmux launches bash non-interactively, which skips ~/.bashrc. Source
# the env file written by prepare_tpu.sh to pick up PATH (uv) and
# LIBTPU_INIT_ARGS (XLA async collective flags).
tmux new-session -d -s "$SESSION" -c "$REPO_DIR" \
    "source '$ENV_FILE' && \
     uv run nanodiffusion pretrain --config $CONFIG \
         --wandb-project $WANDB_PROJECT --wandb-entity $WANDB_ENTITY \
         $RESUME_FLAG; \
     exec bash"

echo ""
echo "Pretrain launched under tmux session '$SESSION'."
echo "  Attach:     tmux attach -t $SESSION"
echo "  Detach:     Ctrl-a d"
echo "  Kill run:   tmux kill-session -t $SESSION"
