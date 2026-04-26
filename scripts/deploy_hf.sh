#!/usr/bin/env bash
# scripts/deploy_hf.sh — Push SENTINEL to HF Space cleanly.
#
# Two known issues with bare `openenv push`:
#   1. It auto-injects `base_path: /web` into the README frontmatter, which
#      breaks HF Spaces' iframe embed (Gradio is mounted at /, not /web).
#   2. It ignores .gitignore and uploads the local venv (env/), training
#      checkpoints, __pycache__/, and unsloth_compiled_cache/ — bloating the
#      Space repo by ~130 MB and polluting the file tree the judges browse.
#
# This wrapper runs `openenv push` and then post-processes the remote Space
# via scripts/hf_post_push_cleanup.py (frontmatter strip, bloat delete, judge
# clutter strip — see script docstring).
# The cleanup lives in a standalone .py file so heredoc parsing differences
# between bash, dash, and Git Bash on Windows don't break the deploy.
set -euo pipefail

REPO_ID="${REPO_ID:-Elliot89/sentinel}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[deploy] openenv push --repo-id $REPO_ID"
PYTHONUTF8=1 openenv push --repo-id "$REPO_ID"

echo "[deploy] running post-push cleanup (frontmatter + bloat)..."
PYTHONUTF8=1 REPO_ID="$REPO_ID" python "$SCRIPT_DIR/hf_post_push_cleanup.py" --repo-id "$REPO_ID"

echo "[deploy] done"
