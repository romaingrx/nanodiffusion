#!/usr/bin/env bash
# Prepare a GCP TPU VM for nanodiffusion training.
#
# Usage:
#   GITHUB_TOKEN=ghp_xxx bash prepare_tpu.sh
#   GITHUB_TOKEN=ghp_xxx bash prepare_tpu.sh --resume   # auto-resume from latest checkpoint
#
# Prerequisites:
#   - GCP TPU VM (v5e, v6e, etc.) with service account access to GCS
#   - For private repo: GITHUB_TOKEN env var

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/romaingrx/nanodiffusion.git}"
REPO_BRANCH="${REPO_BRANCH:-romaingrx/rom-18-scale-training}"
REPO_DIR="${REPO_DIR:-$HOME/nanodiffusion}"
GCS_BUCKET="${GCS_BUCKET:-nanodiffusion-runs}"
GCS_MOUNT="$HOME/gcs-${GCS_BUCKET}"
RESUME="${1:-}"

info() { echo -e "\033[1;34m==>\033[0m $*"; }
warn() { echo -e "\033[1;33m==>\033[0m $*"; }
ok()   { echo -e "\033[1;32m==>\033[0m $*"; }

# --- System packages ---

if ! command -v gcsfuse &> /dev/null; then
    GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | \
        sudo tee /etc/apt/sources.list.d/gcsfuse.list > /dev/null
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
        sudo tee /usr/share/keyrings/cloud.google.asc > /dev/null
fi
sudo apt-get update -qq
sudo apt-get install -y -qq tmux gcsfuse > /dev/null 2>&1
ok "tmux + gcsfuse installed"

cat > ~/.tmux.conf <<'TMUX'
unbind C-b
set -g prefix C-a
bind C-a send-prefix
set -g history-limit 100000
set -g mouse on
TMUX

# --- uv ---

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
grep -q 'HOME/.local/bin' ~/.bashrc 2>/dev/null || \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# --- XLA flags ---
# Async collective fusion overlaps all-reduce with compute.
# JAX_ENABLE_PGLE is NOT set (GPU/CUPTI only, conflicts with --profile-steps).

if ! grep -q 'LIBTPU_INIT_ARGS' ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<'EOF'
export LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
EOF
fi

# --- Repo ---

if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$REPO_BRANCH"
    git pull --ff-only origin "$REPO_BRANCH" || warn "pull failed, using existing state"
else
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        CLONE_URL="https://${GITHUB_TOKEN}@${REPO_URL#https://}"
    else
        CLONE_URL="$REPO_URL"
    fi
    git clone -b "$REPO_BRANCH" "$CLONE_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi
ok "Repo at $(git log --oneline -1)"

cd "$REPO_DIR"
uv sync --all-extras --all-groups
ok "Python deps synced"

# --- GCS mount ---
# All checkpoints and data live on gs://$GCS_BUCKET.
# gcsfuse mounts it locally; symlinks wire it into the repo layout.

if ! mountpoint -q "$GCS_MOUNT" 2>/dev/null; then
    mkdir -p "$GCS_MOUNT"
    gcsfuse --implicit-dirs "$GCS_BUCKET" "$GCS_MOUNT"
fi
ok "GCS mounted at $GCS_MOUNT"

DATA_DIR="${REPO_DIR}/data"
RUNS_DIR="${REPO_DIR}/runs"

# Data symlink
[ -L "$DATA_DIR" ] && rm "$DATA_DIR"
[ ! -e "$DATA_DIR" ] && ln -s "$GCS_MOUNT/data" "$DATA_DIR"

# Runs symlink — checkpoints write directly to GCS so they survive preemption
[ -L "$RUNS_DIR" ] && rm "$RUNS_DIR"
if [ -d "$RUNS_DIR" ]; then
    warn "$RUNS_DIR is a real directory; backing up to ${RUNS_DIR}.local"
    mv "$RUNS_DIR" "${RUNS_DIR}.local"
fi
[ ! -e "$RUNS_DIR" ] && ln -s "$GCS_MOUNT" "$RUNS_DIR"
ok "data -> $GCS_MOUNT/data, runs -> $GCS_MOUNT"

# --- Verify JAX ---

uv run python -c "
import jax
assert jax.default_backend() == 'tpu', f'Expected tpu, got {jax.default_backend()}'
print(f'{len(jax.devices())} TPU chips OK')
"

# --- Auto-resume ---

RESUME_FLAG=""
if [ "$RESUME" = "--resume" ]; then
    # gcsfuse doesn't support symlinks, so find the latest step_* dir
    # across all pretrain runs by sorting numerically.
    LATEST=$(find "$RUNS_DIR/pretrain" -maxdepth 2 -name "step_*" -type d 2>/dev/null \
        | sort -t_ -k2 -n | tail -1)
    if [ -n "$LATEST" ]; then
        RESUME_FLAG="--resume-from $LATEST"
        ok "Will resume from $LATEST"
    else
        warn "No checkpoint found, starting fresh"
    fi
fi

echo ""
ok "TPU VM ready."
echo ""
echo "  tmux new -s nano"
echo "  cd $REPO_DIR"
echo "  uv run nanodiffusion pretrain --config configs/medium.yaml $RESUME_FLAG"
echo ""
