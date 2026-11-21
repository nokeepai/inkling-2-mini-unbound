"""Bake a candidate into a full drop-in checkpoint.

  python3 bake.py <cand_dir> <orig_dir> <out_dir> [--readme PATH] [--build-cand PATH]

Copies every file of <orig_dir> (all shards, mtp, configs, tokenizer, chat template, README) except the HF
download cache, then overwrites ONLY the candidate's tensors in place inside the copied shard(s). Because each
replacement has the same dtype and shape as the original, the safetensors header (names, order, offsets,
metadata) stays byte-identical, and so does the index json. The original directory is never opened for writing.

The candidate's BF16 tensors come from <cand_dir>/tensors.safetensors if present, else they are built from
lowrank.safetensors by harness/build_cand.py (single-thread fp32 CPU, the canonical recipe) and checked against
<cand_dir>/sha256.txt when that file exists.
Writes <out_dir>/../<name>.bake_log.json (outside the checkpoint so the checkpoint has only model files).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import sys
import time

import torch
from safetensors.torch import load_file, save_file

SKIP_NAMES = {".cache"}  # HF hub download cache inside the snapshot dir; not part of the model


def read_header(path: str) -> tuple[int, dict]:
    """(data_start, header) of a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return 8 + n, header


def load_build_cand(path: str):
    spec = importlib.util.spec_from_file_location("build_cand", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidate_tensors(cand: str, orig: str, build_cand_path: str) -> tuple[dict[str, torch.Tensor], str]:
    """Replacement tensors and how they were obtained. Built via build_cand so bytes match the fleet's."""
    full = os.path.join(cand, "tensors.safetensors")
    if os.path.exists(full):
        return load_file(full), "tensors.safetensors"
    bc = load_build_cand(build_cand_path)
    parts = bc.parse(load_file(os.path.join(cand, "lowrank.safetensors")))
    tensors = bc.build(parts, orig)
    canon = os.path.join(cand, "sha256.txt")
    if os.path.exists(canon):
        tmp = os.path.join(cand, ".bake_check.safetensors")
        # sha256.txt hashes save_file's output, so re-serialise exactly as build_cand does
        save_file(tensors, tmp)
        got = bc.sha256(tmp)
        os.remove(tmp)
        want = open(canon).read().split()[0]
        if got != want:
            raise SystemExit(f"built tensors sha {got} != canonical {want}: refusing to bake")
        return tensors, f"built from lowrank, sha256 {got} == canonical"
    return tensors, "built from lowrank (no canonical sha to check)"


def copy_tree(orig: str, out: str) -> list[str]:
    copied = []
    for root, dirs, files in os.walk(orig):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_NAMES)
        rel_root = os.path.relpath(root, orig)
        os.makedirs(os.path.join(out, rel_root), exist_ok=True)
        for fn in sorted(files):
            src, rel = os.path.join(root, fn), os.path.normpath(os.path.join(rel_root, fn))
            shutil.copyfile(src, os.path.join(out, rel))  # real copy, never a hardlink: we write into it
            copied.append(rel)
    return copied


def tensor_file_map(out: str) -> dict[str, str]:
    idx = os.path.join(out, "model.safetensors.index.json")
    if os.path.exists(idx):
        return {k: os.path.join(out, v) for k, v in json.load(open(idx))["weight_map"].items()}
    single = os.path.join(out, "model.safetensors")
    return {k: single for k in read_header(single)[1] if k != "__metadata__"}


def patch_in_place(out: str, tensors: dict[str, torch.Tensor]) -> list[dict]:
    files = tensor_file_map(out)
    done = []
    for name in sorted(tensors):
        t = tensors[name]
        if name not in files:
            raise SystemExit(f"{name}: not in the checkpoint index")
        path = files[name]
        start, header = read_header(path)
        meta = header[name]
        if meta["dtype"] != "BF16" or t.dtype != torch.bfloat16:
            raise SystemExit(f"{name}: only BF16 replacements allowed ({meta['dtype']} / {t.dtype})")
        if list(t.shape) != meta["shape"]:
            raise SystemExit(f"{name}: shape {list(t.shape)} != checkpoint {meta['shape']}")
        b, e = meta["data_offsets"]
        data = t.contiguous().view(torch.uint8).numpy().tobytes()
        if len(data) != e - b:
            raise SystemExit(f"{name}: {len(data)} bytes != slot {e - b}")
        with open(path, "r+b") as f:
            f.seek(start + b)
            old = f.read(e - b)
            f.seek(start + b)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            f.seek(start + b)
            if f.read(e - b) != data:
                raise SystemExit(f"{name}: read-back mismatch")
        done.append({"tensor": name, "file": os.path.relpath(path, out), "bytes": e - b,
                     "changed": old != data,
                     "sha256_orig": hashlib.sha256(old).hexdigest(),
                     "sha256_new": hashlib.sha256(data).hexdigest()})
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cand"), ap.add_argument("orig"), ap.add_argument("out")
    ap.add_argument("--readme", help="replace README.md with this file (model card)")
    ap.add_argument("--build-cand", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_cand.py"))
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} exists and is not empty")
    if os.path.realpath(a.out).startswith(os.path.realpath(a.orig) + os.sep):
        raise SystemExit("out_dir must not be inside orig_dir")
    t0 = time.time()
    tensors, how = candidate_tensors(a.cand, a.orig, a.build_cand)
    t1 = time.time()
    copied = copy_tree(a.orig, a.out)
    t2 = time.time()
    patched = patch_in_place(a.out, tensors)
    if a.readme:
        shutil.copyfile(a.readme, os.path.join(a.out, "README.md"))
    t3 = time.time()
    spec = json.load(open(os.path.join(a.cand, "spec.json")))
    expect = spec.get("diag", {}).get("tensor_sha256", {})
    mism = [p["tensor"] for p in patched if p["tensor"] in expect and expect[p["tensor"]] != p["sha256_new"]]
    log = {"cand_id": spec.get("cand_id"), "cand_dir": os.path.abspath(a.cand), "orig_dir": os.path.abspath(a.orig),
           "out_dir": os.path.abspath(a.out), "source": how, "files_copied": copied, "patched": patched,
           "unchanged_replacements": [p["tensor"] for p in patched if not p["changed"]],
           "spec_tensor_sha_mismatch": mism, "readme_replaced": bool(a.readme),
           "seconds": {"build": round(t1 - t0, 1), "copy": round(t2 - t1, 1), "patch": round(t3 - t2, 1)},
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    log_path = os.path.join(os.path.dirname(os.path.abspath(a.out)), os.path.basename(os.path.abspath(a.out)) + ".bake_log.json")
    json.dump(log, open(log_path, "w"), indent=1)
    print(json.dumps({k: log[k] for k in ("cand_id", "source", "seconds", "spec_tensor_sha_mismatch", "unchanged_replacements")}))
    print(f"patched {len(patched)} tensors; log {log_path}")
    if mism:
        sys.exit(4)


if __name__ == "__main__":
    main()
