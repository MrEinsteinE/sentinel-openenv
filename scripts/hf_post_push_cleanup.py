"""scripts/hf_post_push_cleanup.py — Post-process an HF Space after `openenv push`.

Two known issues with bare `openenv push`:
  1. It auto-injects `base_path: /web` into the README frontmatter, which
     breaks HF Spaces' iframe embed (Gradio is mounted at /, not /web).
  2. It ignores .gitignore and uploads the local venv (env/), training
     checkpoints, __pycache__/, and unsloth_compiled_cache/ — bloating the
     Space repo by ~130 MB and polluting the file tree judges browse.

This script:
  - Strips the injected `base_path:` line from the remote README.md frontmatter.
  - Deletes known bloat folders and __pycache__/.pyc stragglers from the Space.

Run as:
  python scripts/hf_post_push_cleanup.py [--repo-id Elliot89/sentinel]

Env vars:
  HF_TOKEN — required if not logged in via `hf auth login`
  REPO_ID  — overrides --repo-id default
"""

from __future__ import annotations

import argparse
import os
import sys


def fix_frontmatter(api, repo_id: str) -> None:
    from huggingface_hub import hf_hub_download

    print(f"[cleanup] fixing frontmatter on {repo_id}...")
    path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="space")
    with open(path, encoding="utf-8") as f:
        text = f.read()

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_fm = False
    seen_open = False
    for ln in lines:
        if ln.strip() == "---":
            if not seen_open:
                seen_open, in_fm = True, True
            else:
                in_fm = False
            out.append(ln)
            continue
        if in_fm and ln.lstrip().startswith("base_path:"):
            continue
        out.append(ln)

    new_text = "".join(out)
    if new_text != text:
        api.upload_file(
            path_or_fileobj=new_text.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
            commit_message="fix: strip injected base_path frontmatter (Gradio mounted at /)",
        )
        print(f"[cleanup] README.md patched on {repo_id}")
    else:
        print("[cleanup] README.md already clean")


def strip_bloat(api, repo_id: str) -> None:
    print(f"[cleanup] stripping bloat folders from {repo_id}...")
    files = api.list_repo_files(repo_id, repo_type="space")
    folders = [
        "env",
        "backups",
        "training/outputs",
        "training/checkpoints",
        "training/unsloth_compiled_cache",
        "training/.ipynb_checkpoints",
    ]
    for folder in folders:
        if not any(f.startswith(folder + "/") for f in files):
            continue
        try:
            api.delete_folder(
                path_in_repo=folder,
                repo_id=repo_id,
                repo_type="space",
                commit_message=f"cleanup: drop {folder}/ (gitignored, accidentally pushed)",
            )
            print(f"[cleanup] deleted {folder}/")
        except Exception as e:
            msg = str(e).splitlines()[0][:140]
            print(f"[cleanup] skip {folder}/: {msg}")

    files = api.list_repo_files(repo_id, repo_type="space")
    strays = [f for f in files if "__pycache__/" in f or f.endswith(".pyc")]
    if strays:
        from huggingface_hub import CommitOperationDelete

        ops = [CommitOperationDelete(path_in_repo=f) for f in strays]
        api.create_commit(
            repo_id=repo_id,
            repo_type="space",
            operations=ops,
            commit_message=f"cleanup: drop {len(strays)} __pycache__ stragglers",
        )
        print(f"[cleanup] deleted {len(strays)} __pycache__/.pyc stragglers")
    else:
        print("[cleanup] no __pycache__ stragglers")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("REPO_ID", "Elliot89/sentinel"),
        help="HF Space repo id (default: Elliot89/sentinel or $REPO_ID)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[cleanup] huggingface_hub not installed", file=sys.stderr)
        return 1

    api = HfApi()
    fix_frontmatter(api, args.repo_id)
    strip_bloat(api, args.repo_id)
    print("[cleanup] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
