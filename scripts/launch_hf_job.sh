#!/usr/bin/env bash
# scripts/launch_hf_job.sh — launch the Overseer trainer on HF Jobs.
#
# Prerequisite (one-time): `hf auth login` so HF_TOKEN is available
# implicitly via `-s HF_TOKEN`. For the GitHub push, export GITHUB_TOKEN
# in your local shell (a fine-grained PAT with contents:write on
# MrEinsteinE/sentinel-openenv).
#
# Usage:
#     export GITHUB_TOKEN=ghp_...       # one-time, in your shell
#     bash scripts/launch_hf_job.sh
#
# Override defaults via env vars before invoking, e.g.:
#     FLAVOR=a100-large bash scripts/launch_hf_job.sh
#     STEP200_MIN_REWARD=0.90 bash scripts/launch_hf_job.sh
set -euo pipefail

FLAVOR="${FLAVOR:-l4x1}"
TIMEOUT="${TIMEOUT:-6h}"
SENTINEL_URL="${SENTINEL_URL:-https://elliot89-sentinel.hf.space}"
GIT_REPO="${GIT_REPO:-https://github.com/MrEinsteinE/sentinel-openenv}"
GIT_BRANCH="${GIT_BRANCH:-main}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3-1.7B}"
MODEL_REPO="${MODEL_REPO:-Elliot89/sentinel-overseer-qwen3-1.7b}"
STEP100_MIN_REWARD="${STEP100_MIN_REWARD:-0.05}"
STEP200_MIN_REWARD="${STEP200_MIN_REWARD:-0.85}"

if ! command -v hf >/dev/null 2>&1; then
  echo "[launch] error: 'hf' CLI not found. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "[launch] error: GITHUB_TOKEN is not set in your shell. Export it first:" >&2
  echo "    export GITHUB_TOKEN=ghp_xxx" >&2
  echo "  GitHub PAT must have contents:write on MrEinsteinE/sentinel-openenv." >&2
  exit 1
fi

echo "[launch] flavor=${FLAVOR} timeout=${TIMEOUT}"
echo "[launch] SENTINEL_URL=${SENTINEL_URL}"
echo "[launch] MODEL_REPO=${MODEL_REPO}"
echo "[launch] GIT_REPO=${GIT_REPO} (${GIT_BRANCH})"
echo "[launch] abort thresholds: step100<${STEP100_MIN_REWARD}, step200<${STEP200_MIN_REWARD}"
echo

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/training/grpo_hf_job.py"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[launch] error: ${SCRIPT_PATH} not found. Run from repo root." >&2
  exit 1
fi

exec hf jobs uv run \
  --flavor "${FLAVOR}" \
  --timeout "${TIMEOUT}" \
  -s HF_TOKEN \
  -s "GITHUB_TOKEN=${GITHUB_TOKEN}" \
  -e "SENTINEL_URL=${SENTINEL_URL}" \
  -e "GIT_REPO=${GIT_REPO}" \
  -e "GIT_BRANCH=${GIT_BRANCH}" \
  -e "MODEL_NAME=${MODEL_NAME}" \
  -e "MODEL_REPO=${MODEL_REPO}" \
  -e "STEP100_MIN_REWARD=${STEP100_MIN_REWARD}" \
  -e "STEP200_MIN_REWARD=${STEP200_MIN_REWARD}" \
  "${SCRIPT_PATH}"
