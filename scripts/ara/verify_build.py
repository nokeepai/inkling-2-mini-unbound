"""Pod-side: compare build_cand's BF16 tensors with an independent recomputation of the intended W'.
Intended W' := bf16(diag(r)(W + A Bᵀ)) computed in fp32 on CPU (compact-form definition). Also counts elements where
the GPU fp32 version (what the optimiser saw, up to BF16 rounding) differs, and reports row-norm ratios."""
import hashlib
import json
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file

d = sys.argv[1]
idx = json.load(open("/root/w/orig/model.safetensors.index.json"))["weight_map"]
lr = load_file(f"{d}/lowrank.safetensors")
built = load_file(f"{d}/build/tensors.safetensors")
names = sorted({k.rsplit(".", 1)[0] for k in lr})
torch.set_num_threads(1)
out = {"n": len(names), "exact_cpu": 0, "gpu_diff_elems": 0, "rownorm_ratio_min": 1e9, "rownorm_ratio_max": 0.0,
       "tensor_sha256": {}}
for n in names:
    with safe_open(f"/root/w/orig/{idx[n]}", "pt") as f:
        W = f.get_tensor(n)
    A, B, r = lr[n + ".A"], lr[n + ".B"], lr.get(n + ".r")
    Wn = W.float() + A @ B.T
    if r is not None:
        Wn = Wn * r[:, None]
    Wn = Wn.to(torch.bfloat16)
    out["exact_cpu"] += int(torch.equal(Wn, built[n]))
    g = W.float().cuda() + A.cuda() @ B.cuda().T
    if r is not None:
        g = g * r.cuda()[:, None]
    out["gpu_diff_elems"] += int((g.to(torch.bfloat16).cpu() != built[n]).sum())
    rr = built[n].float().norm(dim=1) / W.float().norm(dim=1)
    out["rownorm_ratio_min"] = min(out["rownorm_ratio_min"], float(rr.min()))
    out["rownorm_ratio_max"] = max(out["rownorm_ratio_max"], float(rr.max()))
    out["tensor_sha256"][n] = hashlib.sha256(built[n].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
json.dump(out, open(f"{d}/verify.json", "w"), indent=1)
print("VERIFY", json.dumps({k: v for k, v in out.items() if k != "tensor_sha256"}))
