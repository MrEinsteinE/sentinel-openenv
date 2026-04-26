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
# This wrapper runs `openenv push` and then post-processes the remote Space:
#   - strips the injected `base_path` line so iframes resolve to root
#   - deletes known bloat folders (env/, backups/, training/outputs/,
#     training/unsloth_compiled_cache/, anything matching __pycache__/)
set -euo pipefail

REPO_ID="${REPO_ID:-Elliot89/sentinel}"

echo "[deploy] openenv push --repo-id $REPO_ID"
PYTHONUTF8=1 openenv push --repo-id "$REPO_ID"

echo "[deploy] fixing frontmatter on $REPO_ID..."
PYTHONUTF8=1 python - <<PY
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
path = hf_hub_download(repo_id="$REPO_ID", filename="README.md", repo_type="space")
with open(path, encoding="utf-8") as f:
    text = f.read()
# Strip any `base_path: ...` line inside the YAML frontmatter
lines = text.splitlines(keepends=True)
out, in_fm, seen_open = [], False, False
for ln in lines:
    if ln.strip() == "---":
        if not seen_open:
            seen_open, in_fm = True, True
        else:
            in_fm = False
        out.append(ln)
        continue
    if in_fm and ln.lstrip().startswith("base_path:"):
        continue  # drop
    out.append(ln)
new_text = "".join(out)
if new_text != text:
    api.upload_file(
        path_or_fileobj=new_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id="$REPO_ID",
        repo_type="space",
        commit_message="fix: strip injected base_path frontmatter (Gradio mounted at /)",
    )
    print("[deploy] README.md patched on $REPO_ID")
else:
    print("[deploy] README.md already clean")
PY

echo "[deploy] stripping bloat (env/, backups/, training/outputs/, caches)..."
PYTHONUTF8=1 python - <<PY
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("$REPO_ID", repo_type="space")
folders = ["env", "backups", "training/outputs", "training/unsloth_compiled_cache",
           "training/.ipynb_checkpoints"]
for folder in folders:
    if not any(f.startswith(folder + "/") for f in files):
        continue
    try:
        api.delete_folder(
            path_in_repo=folder,
            repo_id="$REPO_ID",
            repo_type="space",
            commit_message=f"cleanup: drop {folder}/ (gitignored, accidentally pushed)",
        )
        print(f"[deploy] deleted {folder}/")
    except Exception as e:
        print(f"[deploy] skip {folder}/: {str(e)[:120]}")

# Stragglers: any remaining __pycache__/* and *.pyc files
files = api.list_repo_files("$REPO_ID", repo_type="space")
strays = [f for f in files if "__pycache__/" in f or f.endswith(".pyc")]
if strays:
    from huggingface_hub import CommitOperationDelete
    ops = [CommitOperationDelete(path_in_repo=f) for f in strays]
    api.create_commit(
        repo_id="$REPO_ID",
        repo_type="space",
        operations=ops,
        commit_message=f"cleanup: drop {len(strays)} __pycache__ stragglers",
    )
    print(f"[deploy] deleted {len(strays)} __pycache__/.pyc stragglers")
else:
    print("[deploy] no __pycache__ stragglers")
PY

echo "[deploy] done"
