"""Reproduce the edited checkpoint from the original Inkling-2-Mini files and edit.pt.

  python3 apply_edit.py <edit.pt> <orig_dir> <out_dir>

Copies orig_dir to out_dir and overwrites the edited BF16 tensors in place (headers, tensor order and every other
tensor stay byte-identical). Each tensor is computed exactly as the edit was built:
    W' = bf16( diag(r) * (W.float() + A @ B.T) )   fp32, CPU, one thread
and checked against the sha256 stored in edit.pt.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys

import torch
from safetensors import safe_open


def header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    edit_path, orig, out = sys.argv[1:]
    edit = torch.load(edit_path, weights_only=False, map_location="cpu")
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty")
    shutil.copytree(orig, out, ignore=shutil.ignore_patterns(".cache"))
    wmap = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    torch.set_num_threads(1)
    for name, t in sorted(edit["tensors"].items()):
        path = os.path.join(out, wmap[name])
        with safe_open(path, "pt") as f:
            W = f.get_tensor(name)
        X = W.float() + t["A"] @ t["B"].T
        if "r" in t:
            X = X * t["r"][:, None]
        data = X.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
        if hashlib.sha256(data).hexdigest() != t["sha256_edited"]:
            raise SystemExit(f"{name}: rebuilt tensor does not match edit.pt's sha256")
        start, h = header(path)
        b, e = h[name]["data_offsets"]
        assert e - b == len(data), name
        with open(path, "r+b") as f:
            f.seek(start + b)
            f.write(data)
        print(f"edited {name}")
    print(f"done: {len(edit['tensors'])} tensors replaced in {out}")


if __name__ == "__main__":
    main()
