#!/usr/bin/env bash
# Launch pretrain on this TPU VM under tmux (survives ssh disconnect).
#
# Usage (fresh run, first time or after VM recreate):
#   GITHUB_TOKEN=ghp_xxx GCS_BUCKET=nanodiffusion-runs-us-east5 bash launch.sh
#
# Usage (resume after preemption / manual kill):
#   bash launch.sh --resume
#
# Re-invoking while a run is active replaces it; the prior tmux session is
# killed first so this is safe to call repeatedly.

set -euo pipefail

RESUME="${1:-}"
CONFIG="${CONFIG:-configs/medium.yaml}"
SESSION="${SESSION:-nano}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$HOME/nanodiffusion}"

# Bootstrap: repo sync, deps, GCS mount. Idempotent — cheap to call every time.
bash "$SCRIPT_DIR/prepare_tpu.sh" "$RESUME"

# On --resume, point at the newest step_* dir. gcsfuse can't traverse
# symlinks across a reboot so we sort numerically by step number.
RESUME_FLAG=""
if [ "$RESUME" = "--resume" ]; then
    LATEST=$(find "$REPO_DIR/runs/pretrain" -maxdepth 2 -name "step_*" -type d 2>/dev/null \
        | sort -t_ -k2 -n | tail -1)
    if [ -n "$LATEST" ]; then
        RESUME_FLAG="--resume-from $LATEST"
    fi
fi

# Replace any prior run. Safe because nothing outside this session is writing
# to the same run_dir.
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" \
    "cd $REPO_DIR && uv run nanodiffusion pretrain --config $CONFIG $RESUME_FLAG; exec bash"

echo ""
echo "Pretrain launched under tmux session '$SESSION'."
echo "  Attach:     tmux attach -t $SESSION"
echo "  Detach:     Ctrl-a d"
echo "  Kill run:   tmux kill-session -t $SESSION"
