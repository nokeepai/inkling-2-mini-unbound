"""Pod: factor c0 (OU's edit) into the compact form, build it with build_cand.py, compare with OU bit-for-bit.
Two factorings of B are tried: B = W32ᵀ@Q and B = (Qᵀ@W32)ᵀ. A = −s·Q, no r. Writes /root/fid/lr_c0/ (best one)."""
import json
import os
import shutil
import subprocess
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ORIG, OU = "/root/w/orig/model-00001-of-00001.safetensors", "/root/w/ou/model-00001-of-00001.safetensors"
edit = torch.load("/root/w/ou/edit.pt", map_location="cpu", weights_only=True)
Q, s_attn = edit["Q"].float(), edit["s_attn"].float()
res = {}
with safe_open(ORIG, "pt") as fo, safe_open(OU, "pt") as fu:
    names = [f"model.llm.layers.{i}.attn.wo_ud.weight" for i in range(24) if float(s_attn[i]) != 0]
    W = {n: fo.get_tensor(n) for n in names}
    ou = {n: fu.get_tensor(n) for n in names}
for variant in ("WtQ", "QtW_T"):
    lr = {}
    for n in names:
        i = int(n.split(".")[3])
        s = float(s_attn[i])
        W32 = W[n].float()
        lr[f"{n}.A"] = (-s * Q).contiguous()
        lr[f"{n}.B"] = (W32.T @ Q).contiguous() if variant == "WtQ" else (Q.T @ W32).T.contiguous()
    d = f"/root/fid/lr_test_{variant}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    save_file(lr, f"{d}/lowrank.safetensors")
    r = subprocess.run([sys.executable, "/root/fid/build_cand.py", d, "/root/w/orig", d], capture_output=True, text=True)
    built = load_file(f"{d}/tensors.safetensors")
    diffs = {n: int((built[n].view(torch.int16) != ou[n].view(torch.int16)).sum()) for n in names}
    maxabs = max(float((built[n].float() - ou[n].float()).abs().max()) for n in names)
    res[variant] = {"build_stdout": r.stdout.strip(), "build_rc": r.returncode, "total_diff": sum(diffs.values()),
                    "per_tensor": diffs, "max_abs": maxabs,
                    "lowrank_bytes": os.path.getsize(f"{d}/lowrank.safetensors")}
    print(variant, "differing elements:", sum(diffs.values()), "max|diff|", maxabs, r.stdout.strip(), flush=True)
json.dump(res, open("/root/fid/factor_c0.json", "w"), indent=1)
print("FACTOR_DONE")
