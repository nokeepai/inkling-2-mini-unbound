"""Run harness/build_cand.py on candidates locally and check per-tensor BF16 bytes equal the intended ones (spec diag)."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import load_file

R = os.path.expanduser("~/Code/inkling-abl")
ORIGDIR = f"{R}/methods/mpoa/origdir"
os.makedirs(ORIGDIR, exist_ok=True)
link = f"{ORIGDIR}/model.safetensors"
if not os.path.exists(link):
    os.symlink(f"{R}/methods/mpoa/orig_wo_conv.safetensors", link)

ok_all = True
for cid in sys.argv[1:]:
    d = f"{R}/methods/mpoa/cands/{cid}"
    spec = json.load(open(f"{d}/spec.json"))
    want = spec["diag"]["tensor_sha256"]
    with tempfile.TemporaryDirectory() as out:
        subprocess.run([sys.executable, f"{R}/harness/build_cand.py", d, ORIGDIR, out], check=True, capture_output=True)
        T = load_file(f"{out}/tensors.safetensors")
    bad = [n for n, t in T.items() if hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() != want.get(n)]
    ok = not bad and set(T) == set(want)
    ok_all &= ok
    print(cid, "BITEXACT" if ok else f"MISMATCH {len(bad)} tensors {bad[:3]}", flush=True)
sys.exit(0 if ok_all else 1)
