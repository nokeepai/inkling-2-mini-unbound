"""Range-fetch the 24 router weights (mlp.gate.weight, BF16) of orig @ ed40e7da and write a mini orig dir
R/methods/experts/origdir/model.safetensors = mpoa's wo_ud + attn_sconv tensors + the router weights,
so build_cand.py can build C4 candidates on the Mac. Run with /opt/homebrew/bin/python3.13."""
import json
import os
import struct

import requests
import torch
from safetensors.torch import load_file, save_file

R = os.path.expanduser("~/Code/inkling-abl")
url = ("https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/"
       "ed40e7da625f594fc9c4990f5f66be24873c22d7/model-00001-of-00001.safetensors")
tokp = os.path.expanduser("~/.cache/huggingface/token")
H = {"Authorization": f"Bearer {open(tokp).read().strip()}"} if os.path.exists(tokp) else {}


def rng(a, b):
    r = requests.get(url, headers={**H, "Range": f"bytes={a}-{b}"}, timeout=600)
    r.raise_for_status()
    assert len(r.content) == b - a + 1
    return r.content


n = struct.unpack("<Q", rng(0, 7))[0]
hdr = json.loads(rng(8, 8 + n - 1))
base = 8 + n
out = load_file(f"{R}/methods/mpoa/orig_wo_conv.safetensors")
for i in range(24):
    nm = f"model.llm.layers.{i}.mlp.gate.weight"
    e = hdr[nm]
    assert e["dtype"] == "BF16", (nm, e)
    a, b = e["data_offsets"]
    out[nm] = torch.frombuffer(bytearray(rng(base + a, base + b - 1)), dtype=torch.bfloat16).reshape(e["shape"]).clone()
os.makedirs(f"{R}/methods/experts/origdir", exist_ok=True)
save_file(out, f"{R}/methods/experts/origdir/model.safetensors")
print("wrote", len(out), "tensors")
