"""Per-tensor sha256 manifest of base vs patched GGUF, plus KV comparison.  Runs ON the pod.

  python3 verify_patch.py <base.gguf> <patched.gguf> <built tensors.safetensors> <name_map.json> <out_dir>

Writes <out_dir>/tensor_manifest.tsv (name, type, shape, sha_base, sha_patched, same) and patch_proof.json.
Exits non-zero unless exactly the 20 mapped tensors differ, each equals the built edit, and only general.name changed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from gguf import GGUFReader
from safetensors.torch import load_file


def tensor_shas(path: str) -> dict[str, tuple[str, str, list[int]]]:
    rd = GGUFReader(path)
    return {t.name: (hashlib.sha256(np.asarray(t.data).tobytes()).hexdigest(), t.tensor_type.name, [int(x) for x in t.shape])
            for t in rd.tensors}


def kvs(path: str) -> dict[str, str]:
    rd = GGUFReader(path)
    out = {}
    for k, f in rd.fields.items():
        # hash type words and value parts (skip key parts 0-1) so a type or value change shows up
        out[k] = hashlib.sha256(b"".join(np.asarray(p).tobytes() for p in f.parts[2:])).hexdigest()
    return out


def main() -> None:
    base, patched, built, map_path, out_dir = sys.argv[1:]
    name_map: dict[str, str] = json.load(open(map_path))
    with ProcessPoolExecutor(2) as ex:
        fb, fp = ex.submit(tensor_shas, base), ex.submit(tensor_shas, patched)
        sb, sp = fb.result(), fp.result()
    assert list(sb) == list(sp), "tensor list/order differs"
    edited = load_file(built)
    want = {name_map[ck]: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for ck, t in edited.items()}
    rows, differ = [], []
    for n in sb:
        same = sb[n][0] == sp[n][0]
        if not same:
            differ.append(n)
        rows.append(f"{n}\t{sb[n][1]}\t{'x'.join(map(str, sb[n][2]))}\t{sb[n][0]}\t{sp[n][0]}\t{int(same)}")
    open(f"{out_dir}/tensor_manifest.tsv", "w").write("name\ttype\tshape\tsha256_base\tsha256_patched\tsame\n" + "\n".join(rows) + "\n")
    kb, kp = kvs(base), kvs(patched)
    kv_diff = sorted(k for k in set(kb) | set(kp) if kb.get(k) != kp.get(k))
    edits_match = all(sp[g][0] == h for g, h in want.items())
    proof = {"n_tensors": len(sb), "n_differ": len(differ), "differ": differ,
             "differ_equals_mapped_set": sorted(differ) == sorted(want), "patched_equals_built_edit": edits_match,
             "kv_keys_differ": kv_diff, "name_map": name_map}
    json.dump(proof, open(f"{out_dir}/patch_proof.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in proof.items() if k != "name_map"}, indent=1))
    ok = len(differ) == 20 and proof["differ_equals_mapped_set"] and edits_match and kv_diff == ["general.name"]
    print("PATCH_VERIFIED" if ok else "PATCH_VERIFY_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
