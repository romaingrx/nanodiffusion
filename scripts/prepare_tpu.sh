#!/usr/bin/env bash
# Prepare a fresh GCP TPU VM for nanodiffusion training.
#
# Run on the TPU VM after SSH-ing in:
#   bash <(curl -sL https://raw.githubusercontent.com/romaingrx/nanodiffusion/main/scripts/prepare_tpu.sh)
#
# Or copy it manually and run:
#   bash scripts/prepare_tpu.sh
#
# Prerequisites:
#   - A GCP TPU VM (v5e, v6e, etc.)
#   - gs://nanodiffusion-runs must exist with data/ uploaded
#   - For private repo: set GITHUB_TOKEN env var before running

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/romaingrx/nanodiffusion.git}"
REPO_BRANCH="${REPO_BRANCH:-romaingrx/rom-18-scale-training}"
REPO_DIR="${REPO_DIR:-$HOME/nanodiffusion}"
GCS_BUCKET="${GCS_BUCKET:-nanodiffusion-runs}"
DATA_DIR="${REPO_DIR}/data"

info() { echo -e "\033[1;34m==>\033[0m $*"; }
warn() { echo -e "\033[1;33m==>\033[0m $*"; }
ok()   { echo -e "\033[1;32m==>\033[0m $*"; }

info "Installing system packages..."
if ! command -v gcsfuse &> /dev/null; then
    GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | \
        sudo tee /etc/apt/sources.list.d/gcsfuse.list > /dev/null
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
        sudo tee /usr/share/keyrings/cloud.google.asc > /dev/null
fi
sudo apt-get update -qq
sudo apt-get install -y -qq tmux gcsfuse > /dev/null 2>&1
ok "tmux $(tmux -V) + gcsfuse installed"

cat > ~/.tmux.conf <<'TMUX'
unbind C-b
set -g prefix C-a
bind C-a send-prefix
set -g history-limit 100000
set -g mouse on
TMUX
ok "tmux configured (prefix: C-a)"

if ! command -v uv &> /dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
grep -q 'HOME/.local/bin' ~/.bashrc 2>/dev/null || \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
ok "uv $(uv --version)"

# XLA/TPU scheduler flags from the scaling book: async collective fusion
# lets all-gathers overlap with compute, and async all-gather enables the
# gradient all-reduce to run concurrently with the backward pass. Must be
# set via env var because libtpu reads them before jax.config is available.
#
# JAX_ENABLE_PGLE is intentionally *not* set: it's a GPU/CUPTI feature
# (see the "PGLE collected an empty trace" warning it emits on TPU) and
# holds an active profiler session that conflicts with --profile-steps.
if ! grep -q 'LIBTPU_INIT_ARGS' ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<'LIBTPU_FLAGS'
export LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
LIBTPU_FLAGS
fi
ok "LIBTPU_INIT_ARGS exported in ~/.bashrc"

if [ -d "$REPO_DIR/.git" ]; then
    info "Repo exists at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$REPO_BRANCH"
    git pull --ff-only origin "$REPO_BRANCH" || warn "pull failed, using existing state"
else
    info "Cloning $REPO_URL ($REPO_BRANCH)..."
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        CLONE_URL="https://${GITHUB_TOKEN}@${REPO_URL#https://}"
    else
        CLONE_URL="$REPO_URL"
    fi
    git clone -b "$REPO_BRANCH" "$CLONE_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi
ok "Repo ready at $REPO_DIR ($(git log --oneline -1))"

info "Syncing Python deps..."
cd "$REPO_DIR"
uv sync --all-extras --all-groups
ok "Python deps synced"

MOUNT_POINT="$HOME/gcs-${GCS_BUCKET}"
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    warn "Bucket already mounted at $MOUNT_POINT"
else
    info "Mounting gs://$GCS_BUCKET at $MOUNT_POINT..."
    mkdir -p "$MOUNT_POINT"
    gcsfuse --implicit-dirs "$GCS_BUCKET" "$MOUNT_POINT"
fi

if [ -L "$DATA_DIR" ]; then
    rm "$DATA_DIR"
elif [ -d "$DATA_DIR" ]; then
    warn "$DATA_DIR exists and is a real directory, skipping symlink"
fi
if [ ! -e "$DATA_DIR" ]; then
    ln -s "$MOUNT_POINT/data" "$DATA_DIR"
    ok "Data symlinked: $DATA_DIR -> $MOUNT_POINT/data"
else
    warn "Data dir already exists at $DATA_DIR"
fi

info "Verifying JAX TPU backend..."
cd "$REPO_DIR"
uv run python -c "
import jax
backend = jax.default_backend()
devices = jax.devices()
print(f'backend: {backend}')
print(f'devices: {devices}')
assert backend == 'tpu', f'Expected tpu backend, got {backend}'
print('TPU OK')
"
ok "JAX sees $(uv run python -c 'import jax; print(len(jax.devices()))') TPU chip(s)"

echo ""
ok "TPU VM ready. Next steps:"
echo "  tmux new -s nano"
echo "  cd $REPO_DIR"
echo "  uv run nanodiffusion pretrain --config configs/medium.yaml"
echo ""
echo "Bucket contents at $MOUNT_POINT:"
ls "$MOUNT_POINT/" 2>/dev/null || warn "bucket listing failed (gcsfuse may need a moment)"
