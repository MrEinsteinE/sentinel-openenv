#!/usr/bin/env bash
# scripts/launch_trained_eval.sh - sister to launch_zeroshot_eval.sh that
# downloads the trained LoRA from MODEL_REPO and runs the held-out eval
# (SENTINEL_TRAINED_EVAL_ONLY=1) with per-turn capture.
#
# Use this AFTER training has completed (LoRA pushed to Hub) when the
# eval_data/baseline_qwen3_1_7b_trained.json file is missing locally.
#
# Usage:
#     export GITHUB_TOKEN="ghp_xxx"
#     bash scripts/launch_trained_eval.sh

set -euo pipefail

FLAVOR="${FLAVOR:-l4x1}"
TIMEOUT="${TIMEOUT:-2h}"
SENTINEL_URL="${SENTINEL_URL:-https://elliot89-sentinel.hf.space}"
GIT_REPO="${GIT_REPO:-https://github.com/MrEinsteinE/sentinel-openenv}"
GIT_BRANCH="${GIT_BRANCH:-main}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3-1.7B}"
MODEL_REPO="${MODEL_REPO:-Elliot89/sentinel-overseer-qwen3-1.7b}"

HF_BIN=""
if command -v hf >/dev/null 2>&1; then
    HF_BIN="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_BIN="huggingface-cli"
else
    HF_BIN="$(python -c 'import shutil,sys; print(shutil.which("hf") or shutil.which("huggingface-cli") or "", end="")' 2>/dev/null || true)"
fi
if [ -z "$HF_BIN" ]; then
    echo "[launch] error: neither 'hf' nor 'huggingface-cli' on PATH." >&2
    echo "  Install with: pip install -U 'huggingface_hub>=0.27'" >&2
    exit 1
fi

if ! "$HF_BIN" auth whoami >/dev/null 2>&1; then
    echo "[launch] error: not logged in to Hugging Face." >&2
    echo "  Run: $HF_BIN auth login --token hf_xxx --add-to-git-credential" >&2
    exit 1
fi

if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "[launch] error: GITHUB_TOKEN is not set." >&2
    echo "  The PAT must have contents:write on MrEinsteinE/sentinel-openenv." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$REPO_ROOT/training/grpo_hf_job.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[launch] error: $SCRIPT_PATH not found." >&2
    exit 1
fi

echo "[launch] mode=TRAINED-EVAL-ONLY"
echo "[launch] flavor=$FLAVOR timeout=$TIMEOUT"
echo "[launch] SENTINEL_URL=$SENTINEL_URL"
echo "[launch] MODEL_NAME=$MODEL_NAME  (will load LoRA from $MODEL_REPO)"
echo "[launch] GIT_REPO=$GIT_REPO ($GIT_BRANCH)"
echo ""

exec "$HF_BIN" jobs uv run \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    -s HF_TOKEN \
    -s "GITHUB_TOKEN=$GITHUB_TOKEN" \
    -e "SENTINEL_URL=$SENTINEL_URL" \
    -e "GIT_REPO=$GIT_REPO" \
    -e "GIT_BRANCH=$GIT_BRANCH" \
    -e "MODEL_NAME=$MODEL_NAME" \
    -e "MODEL_REPO=$MODEL_REPO" \
    -e "SENTINEL_TRAINED_EVAL_ONLY=1" \
    -e "VLLM_USE_V1=0" \
    "$SCRIPT_PATH"
