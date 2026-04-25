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

# Modern huggingface_hub (>=0.27) ships `hf`; older versions only ship the
# now-deprecated `huggingface-cli`. Prefer `hf`, fall back transparently.
#
# On Windows, `bash` (Git Bash / MSYS) can fail to resolve .exe shims from a
# venv whose path contains spaces, even when the same venv works fine in
# PowerShell. If POSIX lookup fails, ask Python's PATHEXT-aware shutil.which.
HF_CLI=""
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
elif command -v python >/dev/null 2>&1; then
  HF_CLI="$(python -c 'import shutil,sys; sys.stdout.write(shutil.which("hf") or shutil.which("huggingface-cli") or "")' 2>/dev/null || true)"
elif command -v python3 >/dev/null 2>&1; then
  HF_CLI="$(python3 -c 'import shutil,sys; sys.stdout.write(shutil.which("hf") or shutil.which("huggingface-cli") or "")' 2>/dev/null || true)"
fi

if [[ -z "${HF_CLI}" ]]; then
  echo "[launch] error: cannot locate 'hf' or 'huggingface-cli' on PATH." >&2
  echo "  Install with: pip install -U 'huggingface_hub>=0.27'" >&2
  echo "  On Windows, prefer launching natively in PowerShell:" >&2
  echo "      ./scripts/launch_hf_job.ps1" >&2
  echo "  (bash on Windows can drop venv PATH entries that contain spaces.)" >&2
  exit 1
fi

# Confirm we're logged in and surface the username early. Catches the common
# 403 case where the token lacks job.write or you're logged in under the
# wrong account. Force UTF-8 so '✓' in newer hf output doesn't crash on
# non-UTF-8 locales.
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONUTF8="${PYTHONUTF8:-1}"
if ! HF_WHOAMI="$("${HF_CLI}" auth whoami 2>&1)"; then
  echo "[launch] error: not logged in to Hugging Face." >&2
  echo "  Run: ${HF_CLI} auth login --token hf_xxx --add-to-git-credential" >&2
  echo "  (HF Jobs needs job.write — generate a Write-scope token at" >&2
  echo "   https://huggingface.co/settings/tokens)" >&2
  exit 1
fi

# Parse username from one of two known formats:
#   newer:  "✓ Logged in\n  user: Elliot89"
#   older:  "Elliot89"
HF_USER="$(printf "%s\n" "${HF_WHOAMI}" | awk -F'[[:space:]]+' '/^[[:space:]]*user:/ {print $NF; exit}')"
if [[ -z "${HF_USER}" ]]; then
  HF_USER="$(printf "%s\n" "${HF_WHOAMI}" | grep -v '^[[:space:]]*$' | tail -n1 | tr -d '[:space:]')"
fi

EXPECTED_NS="${MODEL_REPO%%/*}"
if [[ "${HF_USER}" != "${EXPECTED_NS}" ]]; then
  echo "[launch] warning: logged in as '${HF_USER}' but MODEL_REPO targets namespace '${EXPECTED_NS}'." >&2
  echo "  The HF Job will run under '${HF_USER}'. Pushing the adapter to '${MODEL_REPO}'" >&2
  echo "  will 403 unless that account has write access there." >&2
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "[launch] error: GITHUB_TOKEN is not set in your shell." >&2
  echo "  Export it first, e.g.:" >&2
  echo "      export GITHUB_TOKEN=ghp_xxx" >&2
  echo "  The PAT must have contents:write on MrEinsteinE/sentinel-openenv." >&2
  exit 1
fi

echo "[launch] flavor=${FLAVOR} timeout=${TIMEOUT}"
echo "[launch] SENTINEL_URL=${SENTINEL_URL}"
echo "[launch] MODEL_REPO=${MODEL_REPO}"
echo "[launch] GIT_REPO=${GIT_REPO} (${GIT_BRANCH})"
echo "[launch] abort thresholds: step100<${STEP100_MIN_REWARD}, step200<${STEP200_MIN_REWARD}"
echo "[launch] CLI=${HF_CLI}"
echo "[launch] hf user=${HF_USER}"
echo

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/training/grpo_hf_job.py"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[launch] error: ${SCRIPT_PATH} not found. Run from repo root." >&2
  exit 1
fi

exec "${HF_CLI}" jobs uv run \
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
  -e "VLLM_USE_V1=0" \
  "${SCRIPT_PATH}"
