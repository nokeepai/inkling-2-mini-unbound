#!/usr/bin/env python3
"""Push the model card (README.md), charts/ and abliteration/ small files to the PRIVATE HF repo from the Mac.
Weights are untouched (already uploaded from the pod). Refuses if the repo is not private."""
import glob, os
from huggingface_hub import HfApi, CommitOperationAdd

REPO = "flashvenom/Inkling-2-Mini-Unbound"
R = "/Users/nikshepsvn/Code/inkling-abl"
api = HfApi()
assert api.repo_info(REPO).private, "repo is not private; refusing"
ops = [CommitOperationAdd("README.md", f"{R}/out/final/MODEL_CARD.md")]
for p in sorted(glob.glob(f"{R}/package/charts/*.png")):
    ops.append(CommitOperationAdd(f"charts/{os.path.basename(p)}", p))
for name in ("edit.pt", "apply_edit.py", "MANIFEST.json", "MANIFEST.md"):
    p = f"{R}/out/final/{name}"
    if os.path.exists(p):
        ops.append(CommitOperationAdd(f"abliteration/{name}", p))
for p in sorted(glob.glob(f"{R}/out/final/dora/*")):
    ops.append(CommitOperationAdd(f"abliteration/dora/{os.path.basename(p)}", p))
c = api.create_commit(REPO, operations=ops, commit_message="Model card, charts, reproduction files")
print("pushed", len(ops), "files:", c.commit_url)
print("private:", api.repo_info(REPO).private)
