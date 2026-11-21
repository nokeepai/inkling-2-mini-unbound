"""Leak and weight change of partial conv fixes (svd1..svd3) vs plain/exact4 per layer, s=1, no MP."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import mpoa_lib as M

o = M.orig()
Q, _ = M.basis({"r": 2})
print("layer " + " ".join(f"{m:>14s}" for m in ("plain", "svd1", "svd2", "svd3", "exact4")))
for l in M.LAYERS:
    row = []
    for mode in ("plain", "svd1", "svd2", "svd3", "exact4"):
        U = M.subspace(Q, l, mode)
        e = M.edit_layer(l, U, 1.0, "none")
        Wn = M.apply_lowrank(o[M.wname(l)], e["A"], e["B"], None)
        dw = float((Wn.float() - o[M.wname(l)].float()).norm() / o[M.wname(l)].float().norm())
        row.append(f"{M.leak(Wn, l, Q):.3f}/{dw:.4f}")
    print(f"{l:5d} " + " ".join(f"{x:>14s}" for x in row))
