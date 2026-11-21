"""Run on ink-research-1 (CPU only). Compares orig vs OU checkpoints tensor by tensor and checks that
W' = W - s Q Q^T W on attn.wo_ud reproduces OU's tensors. Writes JSON to /root/logs/recon_verify.json."""
import hashlib
import json
import struct
import sys
import time

import torch
from safetensors import safe_open

ORIG = "/root/w/orig/model-00001-of-00001.safetensors"
OU = "/root/w/ou/model-00001-of-00001.safetensors"
EDIT = "/root/w/ou/edit.pt"
OUT = "/root/logs/recon_verify.json"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def tensor_hashes(path):
    header, base = read_header(path)
    meta = header.pop("__metadata__", None)
    hashes = {}
    with open(path, "rb") as f:
        for name, info in header.items():
            start, end = info["data_offsets"]
            f.seek(base + start)
            h = hashlib.blake2b(digest_size=16)
            remaining = end - start
            while remaining:
                chunk = f.read(min(remaining, 1 << 26))
                h.update(chunk)
                remaining -= len(chunk)
            hashes[name] = h.hexdigest()
    return header, meta, hashes


t0 = time.time()
h_orig, meta_orig, hash_orig = tensor_hashes(ORIG)
print("orig hashed", round(time.time() - t0), flush=True)
h_ou, meta_ou, hash_ou = tensor_hashes(OU)
print("ou hashed", round(time.time() - t0), flush=True)

only_orig = sorted(set(h_orig) - set(h_ou))
only_ou = sorted(set(h_ou) - set(h_orig))
meta_diff = sorted(
    n for n in set(h_orig) & set(h_ou)
    if (h_orig[n]["dtype"], h_orig[n]["shape"]) != (h_ou[n]["dtype"], h_ou[n]["shape"])
)
differ = sorted(n for n in set(h_orig) & set(h_ou) if hash_orig[n] != hash_ou[n])
print("differ", len(differ), flush=True)

edit = torch.load(EDIT, map_location="cpu", weights_only=True)
Q = edit["Q"]
s_attn = edit["s_attn"].tolist()
checks = []
with safe_open(ORIG, "pt") as fo, safe_open(OU, "pt") as fu:
    for layer, s in enumerate(s_attn):
        name = f"model.llm.layers.{layer}.attn.wo_ud.weight"
        W = fo.get_tensor(name)
        Wou = fu.get_tensor(name)
        row = {"layer": layer, "s": s, "dtype": str(W.dtype), "shape": list(W.shape)}
        for label, dt in (("fp32", torch.float32), ("fp64", torch.float64)):
            Wd, Qd = W.to(dt), Q.to(dt)
            pred = (Wd - s * Qd @ (Qd.T @ Wd)).to(W.dtype)
            diff = (pred.float() - Wou.float()).abs()
            row[label] = {
                "max_abs_diff_bf16": diff.max().item(),
                "n_mismatch": int((diff > 0).sum().item()),
                "numel": diff.numel(),
            }
        delta = (Wou.float() - W.float())
        # how much of the delta lies in span(Q)
        in_span = Q.float() @ (Q.float().T @ delta)
        row["delta_norm"] = delta.norm().item()
        row["delta_in_spanQ_frac"] = (in_span.norm() / (delta.norm() + 1e-30)).item()
        row["W_norm"] = W.float().norm().item()
        row["QtW_norm"] = (Q.float().T @ W.float()).norm().item()
        row["QtWou_norm"] = (Q.float().T @ Wou.float()).norm().item()
        row["row_norm_change_max_rel"] = (
            (Wou.float().norm(dim=1) - W.float().norm(dim=1)).abs() / W.float().norm(dim=1)
        ).max().item()
        checks.append(row)
        print(layer, s, row["fp32"], flush=True)

result = {
    "n_tensors_orig": len(h_orig),
    "n_tensors_ou": len(h_ou),
    "only_in_orig": only_orig,
    "only_in_ou": only_ou,
    "dtype_or_shape_differs": meta_diff,
    "tensors_with_different_bytes": differ,
    "metadata_orig": meta_orig,
    "metadata_ou": meta_ou,
    "wo_ud_checks": checks,
    "dtypes_of_differing": {n: h_orig[n]["dtype"] for n in differ},
    "seconds": round(time.time() - t0),
}
with open(OUT, "w") as f:
    json.dump(result, f, indent=1)
print("VERIFY_DONE", flush=True)
