#!/usr/bin/env python3
"""Compare sha256 of README and charts in the private HF repo against the local final files."""
import hashlib, glob, os
from huggingface_hub import HfApi, hf_hub_download

REPO = "flashvenom/Inkling-2-Mini-Unbound"
R = "/Users/nikshepsvn/Code/inkling-abl"
pairs = [("README.md", f"{R}/out/final/MODEL_CARD.md")]
pairs += [(f"charts/{os.path.basename(p)}", p) for p in sorted(glob.glob(f"{R}/package/charts/*.png"))]


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


ok = True
for remote, local in pairs:
    rp = hf_hub_download(REPO, remote, force_download=True)
    match = sha(rp) == sha(local)
    ok &= match
    print(f"{'OK ' if match else 'DIFF'} {remote:28s} hf {sha(rp)} local {sha(local)}")
print("ALL MATCH" if ok else "MISMATCH", "| private:", HfApi().repo_info(REPO).private)
