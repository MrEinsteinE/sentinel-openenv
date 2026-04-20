#!/usr/bin/env bash
# scripts/deploy_hf.sh — Push SENTINEL to HF Space cleanly.
#
# The openenv CLI auto-injects `base_path: /web` into the README frontmatter,
# which breaks HF Spaces' iframe embed. This wrapper runs `openenv push`
# and then uses the HF API to strip the injected `base_path` line from the
# remote README so HF iframes the root path (where our Gradio is mounted).
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

echo "[deploy] done"
