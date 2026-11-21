"""Fetch only the 20 wo_ud + 24 attn_sconv tensors of orig Inkling-2-Mini via HTTP range requests.
Writes R/methods/mpoa/orig_wo_conv.safetensors (BF16, exact checkpoint names)."""
import json
import os
import struct
import sys

import requests
import torch
from safetensors.torch import save_file

REPO = "thinkingmachines/Inkling-2-Mini"
REV = sys.argv[1] if len(sys.argv) > 1 else "main"
FILE = "model-00001-of-00001.safetensors"
OUT = os.path.expanduser("~/Code/inkling-abl/methods/mpoa/orig_wo_conv.safetensors")
tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
H = {"Authorization": f"Bearer {tok}"}
url = f"https://huggingface.co/{REPO}/resolve/{REV}/{FILE}"


def rng(a: int, b: int) -> bytes:
    r = requests.get(url, headers={**H, "Range": f"bytes={a}-{b}"}, timeout=600)
    r.raise_for_status()
    assert len(r.content) == b - a + 1, (len(r.content), a, b)
    return r.content


n = struct.unpack("<Q", rng(0, 7))[0]
hdr = json.loads(rng(8, 8 + n - 1))
base = 8 + n
names = [f"model.llm.layers.{i}.attn.wo_ud.weight" for i in range(4, 24)] + \
        [f"model.llm.layers.{i}.attn_sconv.weight" for i in range(24)]
out = {}
for nm in names:
    e = hdr[nm]
    assert e["dtype"] == "BF16", (nm, e)
    a, b = e["data_offsets"]
    buf = bytearray(rng(base + a, base + b - 1))
    out[nm] = torch.frombuffer(buf, dtype=torch.bfloat16).reshape(e["shape"]).clone()
    print(nm, e["shape"], flush=True)
save_file(out, OUT)
print("wrote", OUT)
