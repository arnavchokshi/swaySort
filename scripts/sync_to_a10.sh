#!/usr/bin/env bash
# Sync the BEST_ID_STRAT codebase + the 9 ground-truthed test clips to the
# Lambda Labs A10 GPU instance, then optionally bootstrap a Python env on it.
#
# Layout on the A10 after a successful run:
#   /home/ubuntu/code/best_id_strat/   <- codebase (this repo, no work/ artifacts)
#   /home/ubuntu/clips/                <- the 9 clip folders (videos + gt/)
#   /home/ubuntu/clips/<clip>/<video>  <- matches configs/clips.remote.example.json
#
# Usage:
#   scripts/sync_to_a10.sh            # sync code + clips, no env install
#   scripts/sync_to_a10.sh code       # sync only the code
#   scripts/sync_to_a10.sh clips      # sync only the clips
#   scripts/sync_to_a10.sh env        # only run the env-install step on the A10
#   scripts/sync_to_a10.sh all        # code + clips + env install (full bootstrap)
#
# Env vars (override defaults):
#   A10_HOST       (default: ubuntu@141.148.49.145)
#   A10_KEY        (default: ~/.ssh/pose-tracking.pem)
#   A10_CODE_DIR   (default: /home/ubuntu/code/best_id_strat)
#   A10_CLIPS_DIR  (default: /home/ubuntu/clips)
#   CLIPS_SRC_ROOT (default: /Users/arnavchokshi/Desktop)
#   A10_ENV_NAME   (default: pose-bench)
#
# This script is idempotent: rsync only ships changed files.

set -euo pipefail

A10_HOST="${A10_HOST:-ubuntu@141.148.49.145}"
A10_KEY="${A10_KEY:-$HOME/.ssh/pose-tracking.pem}"
A10_CODE_DIR="${A10_CODE_DIR:-/home/ubuntu/code/best_id_strat}"
A10_CLIPS_DIR="${A10_CLIPS_DIR:-/home/ubuntu/clips}"
CLIPS_SRC_ROOT="${CLIPS_SRC_ROOT:-/Users/arnavchokshi/Desktop}"
A10_ENV_NAME="${A10_ENV_NAME:-pose-bench}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# All 9 ground-truthed clip directories (must exactly match
# configs/clips.remote.example.json names).
CLIPS=(
  BigTest
  easyTest
  adiTest
  mirrorTest
  gymTest
  loveTest
  MotionTest
  shorterTest
  darkTest
)

SSH_OPTS=(-i "$A10_KEY" -o StrictHostKeyChecking=no)
RSYNC_SSH="ssh -i $A10_KEY -o StrictHostKeyChecking=no"

mode="${1:-default}"

log() { printf '\033[1;34m[sync]\033[0m %s\n' "$*"; }

sync_code() {
  log "syncing code -> ${A10_HOST}:${A10_CODE_DIR}"
  ssh "${SSH_OPTS[@]}" "$A10_HOST" "mkdir -p '$A10_CODE_DIR'"
  # Use a temp filter file to avoid the rsync 3.2.7 "recv_rules
  # buffer overflow" bug we hit when combining many --exclude flags
  # with --delete-excluded over ssh. Filter file is rebuilt each call.
  local filter_file
  filter_file="$(mktemp -t best_id_sync_filter.XXXXXX)"
  trap 'rm -f "$filter_file"' RETURN
  cat >"$filter_file" <<'FILTER'
- .git/
- .venv/
- venv/
- env/
- node_modules/
- __pycache__/
- *.pyc
- .DS_Store
- *.det_cache.pkl
- *.cache.pkl
- *.egg-info/
- *.bak
- configs/clips.json
- work/regression/
- work/sweeps/
- work/results/
- work/research/
FILTER
  rsync -az --stats \
    -e "$RSYNC_SSH" \
    --filter="merge $filter_file" \
    "$REPO_ROOT/" "$A10_HOST:$A10_CODE_DIR/"
  log "code sync done"
}

sync_clips() {
  log "syncing ${#CLIPS[@]} clip folders -> ${A10_HOST}:${A10_CLIPS_DIR}"
  ssh "${SSH_OPTS[@]}" "$A10_HOST" "mkdir -p '$A10_CLIPS_DIR'"
  for clip in "${CLIPS[@]}"; do
    src="$CLIPS_SRC_ROOT/$clip"
    if [[ ! -d "$src" ]]; then
      log "WARNING: clip dir $src not found, skipping"
      continue
    fi
    log "  -> $clip"
    local clip_filter
    clip_filter="$(mktemp -t best_id_clip_filter.XXXXXX)"
    cat >"$clip_filter" <<'FILTER'
- .DS_Store
- *.det_cache.pkl
- *.cache.pkl
- *.bak
- *.zip
- *_overlay.mp4
- *_ids_overlay.mp4
- sam2_compare_*/
- yolo_vis_*/
FILTER
    rsync -az --stats --progress \
      -e "$RSYNC_SSH" \
      --filter="merge $clip_filter" \
      "$src/" "$A10_HOST:$A10_CLIPS_DIR/$clip/"
    rm -f "$clip_filter"
  done
  log "clips sync done"
}

setup_env() {
  log "bootstrapping conda env '$A10_ENV_NAME' on A10"
  # Heredoc is quoted to prevent local expansion of $vars.
  # shellcheck disable=SC2087
  ssh "${SSH_OPTS[@]}" "$A10_HOST" bash -se <<EOF
set -euo pipefail
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh

if conda env list | awk '{print \$1}' | grep -qx '$A10_ENV_NAME'; then
  echo "[env] reusing existing env $A10_ENV_NAME"
else
  echo "[env] creating env $A10_ENV_NAME (python 3.11)"
  conda create -y -n '$A10_ENV_NAME' python=3.11
fi

conda activate '$A10_ENV_NAME'

echo "[env] installing torch 2.4.x cu121 wheel (matches A10 driver 535.x / cu12)"
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121

echo "[env] installing project requirements"
pip install -r '$A10_CODE_DIR/requirements.txt'

echo "[env] sanity check"
python - <<PY
import torch, ultralytics, boxmot, motmetrics, cv2, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "device count", torch.cuda.device_count())
print("ultralytics", ultralytics.__version__,
      "boxmot", boxmot.__version__,
      "motmetrics", motmetrics.__version__)
print("cv2", cv2.__version__, "numpy", numpy.__version__)
PY

echo "[env] bootstrap OK"
EOF
  log "env bootstrap done"
}

case "$mode" in
  code)   sync_code ;;
  clips)  sync_clips ;;
  env)    setup_env ;;
  all)    sync_code; sync_clips; setup_env ;;
  default)
    sync_code
    sync_clips
    log "skipping env install; pass 'all' or 'env' to bootstrap conda env"
    ;;
  *)
    echo "usage: $0 [code|clips|env|all|default]" >&2
    exit 2
    ;;
esac

log "all done; remote code at $A10_HOST:$A10_CODE_DIR, clips at $A10_HOST:$A10_CLIPS_DIR"
