"""Build a candidate's full BF16 tensors from its compact form (harness/CANDIDATE_FORMAT.md §"Compact form").

  python3 build_cand.py <cand_dir> <orig_weights_dir> <out_dir>

Reads <cand_dir>/lowrank.safetensors. For each edited checkpoint tensor N it holds `N.A` [out,k] fp32, `N.B` [in,k] fp32
and optionally `N.r` [out] fp32. On CPU, single thread, in fp32 (fixed matmul order: A @ B.T, then add to W, then scale rows):
    W' = bf16( diag(r) · (W_fp32 + A·Bᵀ) )        (r = 1 if absent; diag(r)·X computed as X * r[:, None])
Writes <out_dir>/tensors.safetensors (keys = exact checkpoint names, BF16, original shapes) and
<out_dir>/built_sha256.txt. sha256.txt is the canonical sha, recorded by the first build (ink-research-2):
- If <out_dir>/sha256.txt is absent, it is written with the built sha (exit 0).
- If present and equal, exit 0. If present and different, exit 3 (the caller fetches the canonical tensors and logs it).
Refuses expert/quantised tensors, non-BF16 originals, shape mismatches and unknown keys.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FORBIDDEN = re.compile(r"experts|w13_weight|w2_weight|_scale|amax|outer_scale")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def weight_files(orig: str) -> dict[str, str]:
    """Checkpoint tensor name -> safetensors file, from the index (or the single file)."""
    idx = os.path.join(orig, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(orig, v) for k, v in wm.items()}
    single = os.path.join(orig, "model.safetensors")
    with safe_open(single, "pt") as f:
        return {k: single for k in f.keys()}


def parse(lowrank: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    parts: dict[str, dict[str, torch.Tensor]] = {}
    for key, t in lowrank.items():
        name, _, part = key.rpartition(".")
        if part not in ("A", "B", "r") or not name:
            raise SystemExit(f"bad key {key!r}: expected <tensor name>.A|.B|.r")
        if t.dtype != torch.float32:
            raise SystemExit(f"{key}: must be float32, got {t.dtype}")
        parts.setdefault(name, {})[part] = t
    for name, p in parts.items():
        if "A" not in p or "B" not in p:
            raise SystemExit(f"{name}: needs both .A and .B")
    return parts


def build(parts: dict[str, dict[str, torch.Tensor]], orig: str) -> dict[str, torch.Tensor]:
    files = weight_files(orig)
    out: dict[str, torch.Tensor] = {}
    torch.set_num_threads(1)  # single thread: fixed reduction order, same sha across runs
    for name in sorted(parts):
        if FORBIDDEN.search(name):
            raise SystemExit(f"{name}: expert/quantised tensors may not be edited")
        if name not in files:
            raise SystemExit(f"{name}: not a checkpoint tensor")
        with safe_open(files[name], "pt") as f:
            W = f.get_tensor(name)
        if W.dtype != torch.bfloat16 or W.dim() != 2:
            raise SystemExit(f"{name}: original must be a 2-D BF16 tensor, got {W.dtype} {tuple(W.shape)}")
        A, B, r = parts[name]["A"], parts[name]["B"], parts[name].get("r")
        out_dim, in_dim = W.shape
        if A.dim() != 2 or B.dim() != 2 or A.shape[0] != out_dim or B.shape[0] != in_dim or A.shape[1] != B.shape[1]:
            raise SystemExit(f"{name}: A {tuple(A.shape)} / B {tuple(B.shape)} do not fit W {tuple(W.shape)}")
        Wn = W.float() + A @ B.T
        if r is not None:
            if r.shape != (out_dim,):
                raise SystemExit(f"{name}: r must be [{out_dim}], got {tuple(r.shape)}")
            Wn = Wn * r[:, None]
        out[name] = Wn.to(torch.bfloat16).contiguous()
    return out


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    cand, orig, out_dir = sys.argv[1:]
    parts = parse(load_file(os.path.join(cand, "lowrank.safetensors")))
    tensors = build(parts, orig)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tensors.safetensors")
    save_file(tensors, path)
    got = sha256(path)
    open(os.path.join(out_dir, "built_sha256.txt"), "w").write(got + "  tensors.safetensors\n")
    canon = os.path.join(out_dir, "sha256.txt")
    if not os.path.exists(canon):
        open(canon, "w").write(got + "  tensors.safetensors\n")
        print(f"built {len(tensors)} tensors; sha256 {got} (recorded as canonical)")
        print(got)
        return
    want = open(canon).read().split()[0]
    if want != got:
        print(f"SHA MISMATCH: built {got} != canonical {want}", file=sys.stderr)
        raise SystemExit(3)
    print(f"built {len(tensors)} tensors; sha256 {got} == canonical")
    print(got)


if __name__ == "__main__":
    main()
