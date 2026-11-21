#!/usr/bin/env python3
"""Verify uploaded GGUF sha256 (HF LFS metadata) against our MANIFEST.tsv, then tag v1.0 on both repos.
Token from env HF_TOKEN (org token), never printed."""
import csv, os
from huggingface_hub import HfApi

a = HfApi(token=os.environ["HF_TOKEN"])
man = {r["file"]: r["sha256"] for r in csv.DictReader(open("/Users/nikshepsvn/Code/inkling-abl/package/gguf/MANIFEST.tsv"), delimiter="\t")}
info = a.repo_info("nokeepai/Inkling-2-Mini-Unbound-GGUF", files_metadata=True)
ok = True
for s in info.siblings:
    if s.rfilename in man:
        hub = s.lfs.sha256 if s.lfs else None
        match = hub == man[s.rfilename]
        ok &= match
        print(("OK  " if match else "DIFF"), s.rfilename, (hub or "")[:12])
print("ALL MATCH" if ok else "MISMATCH")
if ok:
    for repo in ("nokeepai/Inkling-2-Mini-Unbound", "nokeepai/Inkling-2-Mini-Unbound-GGUF"):
        try:
            a.create_tag(repo, tag="v1.0", tag_message="Inkling-2-Mini-Unbound v1.0")
            print("tagged v1.0:", repo)
        except Exception as e:
            print("tag:", repo, type(e).__name__, str(e)[:80])
