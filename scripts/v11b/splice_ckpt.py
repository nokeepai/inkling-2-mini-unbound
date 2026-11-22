"""Build the v2 HF checkpoint by splicing v2's tensors into v1.0's safetensors IN PLACE (same dtype/shape -> same
byte ranges), so every other byte of every file stays identical to v1.0. Verifies before and after.

  python3 splice_ckpt.py <v1_ckpt_dir> <v2_tensors.safetensors> <out_dir>

out_dir gets hard copies of every v1 file; only model-*.safetensors shards that hold a replaced tensor change.
Writes <out_dir>/../v2_splice_report.json (per-tensor sha256 before/after, file sha256).
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

DT = {"BF16": torch.bfloat16}


def header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    src, tp, out = sys.argv[1:4]
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns(".cache"))
    idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    new = {}
    with safe_open(tp, "pt") as f:
        for k in f.keys():
            new[k] = f.get_tensor(k)
    rep = {"tensors": {}, "files": {}}
    by_file: dict[str, list[str]] = {}
    for k in new:
        assert k in idx, f"{k} not in v1 checkpoint"
        by_file.setdefault(idx[k], []).append(k)
    for fn, keys in by_file.items():
        p = f"{out}/{fn}"
        base, hdr = header(p)
        with open(p, "r+b") as f:
            for k in sorted(keys):
                m = hdr[k]
                t = new[k]
                assert m["dtype"] == "BF16" and t.dtype == torch.bfloat16 and list(t.shape) == m["shape"], k
                s, e = m["data_offsets"]
                raw = t.contiguous().view(torch.uint16).numpy().tobytes()
                assert len(raw) == e - s, k
                f.seek(base + s)
                old = f.read(e - s)
                f.seek(base + s)
                f.write(raw)
                rep["tensors"][k] = {"v1_sha256": hashlib.sha256(old).hexdigest(),
                                     "v2_sha256": hashlib.sha256(raw).hexdigest(), "changed": old != raw}
    # verify: every replaced tensor reads back equal; every other tensor byte-identical to v1
    for fn in sorted(set(idx.values())):
        a, b = f"{src}/{fn}", f"{out}/{fn}"
        base_a, hdr_a = header(a)
        base_b, hdr_b = header(b)
        assert hdr_a == hdr_b and base_a == base_b, f"header changed in {fn}"
        with open(a, "rb") as fa, open(b, "rb") as fb:
            for k, m in hdr_a.items():
                if k == "__metadata__":
                    continue
                s, e = m["data_offsets"]
                fa.seek(base_a + s)
                fb.seek(base_b + s)
                ra, rb = fa.read(e - s), fb.read(e - s)
                if k in new:
                    assert rb == new[k].contiguous().view(torch.uint16).numpy().tobytes(), f"readback {k}"
                else:
                    assert ra == rb, f"untouched tensor differs: {k}"
        rep["files"][fn] = {"v1_sha256": sha_file(a), "v2_sha256": sha_file(b)}
    n_changed = sum(v["changed"] for v in rep["tensors"].values())
    rep["summary"] = {"replaced": len(new), "actually_changed": n_changed,
                      "other_files_identical": all(sha_file(f"{src}/{x}") == sha_file(f"{out}/{x}")
                                                   for x in ("mtp.safetensors", "config.json", "tokenizer.json"))}
    json.dump(rep, open(os.path.join(os.path.dirname(out.rstrip("/")), "v2_splice_report.json"), "w"), indent=1)
    print("SPLICE_OK", json.dumps(rep["summary"]))


if __name__ == "__main__":
    main()
