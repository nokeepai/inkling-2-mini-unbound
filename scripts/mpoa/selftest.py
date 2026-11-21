"""Sanity checks for mpoa_lib: orig tensors, leak per conv mode, MP row norms, lowrank round trip."""
import sys
import time

import torch
from safetensors.torch import load_file

sys.path.insert(0, __import__("os").path.dirname(__file__))
import mpoa_lib as M

o = M.orig()
W4 = o[M.wname(4)].float()
print("row norm L4 min/max", float(W4.norm(dim=1).min()), float(W4.norm(dim=1).max()), "fro", float(W4.norm()))
ou = load_file(f"{M.R}/methods/baselines/cands/ou/tensors.safetensors")
dlt = ou[M.wname(12)].float() - o[M.wname(12)].float()
sv = torch.linalg.svdvals(dlt)
print("ou-orig L12 delta top svals", [round(float(x), 4) for x in sv[:4]])

Q, info = M.basis({"r": 2})
print("basis info", info)
for mode in ("plain", "exact5", "exact4"):
    for mp in (False, True):
        U = M.subspace(Q, 6, mode)
        e = M.edit_layer(6, U, 1.0, mp)
        Wn = M.apply_lowrank(o[M.wname(6)], e["A"], e["B"], e["r"])
        rn = Wn.float().norm(dim=1) / o[M.wname(6)].float().norm(dim=1)
        print(f"L6 {mode:6s} mp={mp} k={U.shape[1]} leak={M.leak(Wn, 6, Q):.5f} rownorm ratio {float(rn.min()):.5f}-{float(rn.max()):.5f}")

t = time.time()
p = {"r": 2, "conv": "exact4", "mp": True, "s_list": [1.0] * 20}
lr, diag = M.build(p)
print("build s", round(time.time() - t, 1), "k", diag["k"][4], "ntensors", len(lr))
W = M.materialise(lr)
leaks = [round(M.leak(W[M.wname(l)], l, Q), 5) for l in M.LAYERS]
print("leaks exact4 mp all layers", leaks)
# reflection check: s = 2 should give Q^T y' ~ -Q^T y
p2 = {"r": 2, "conv": "exact4", "mp": False, "s_list": [2.0] * 20}
lr2, _ = M.build(p2)
W2 = M.materialise(lr2)
print("s=2 leak (expect ~1.0, sign flipped)", round(M.leak(W2[M.wname(10)], 10, Q), 4))
