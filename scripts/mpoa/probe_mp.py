"""Compare MP variants per layer for the exact4 subspace: leak and row-norm damage."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import mpoa_lib as M

o = M.orig()
Q, _ = M.basis({"r": int(sys.argv[1]) if len(sys.argv) > 1 else 2})
for l in M.LAYERS:
    W = o[M.wname(l)].float()
    for mode in ("exact4", "plain"):
        U = M.subspace(Q, l, mode)
        X = W - U @ (U.T @ W)
        n0, n1 = W.norm(dim=1), X.norm(dim=1)
        Wp = (n0 / n1)[:, None] * X
        fr = n1 / n0
        print(l, mode, f"leak proj={M.leak(X.bfloat16(), l, Q):.4f} postMP={M.leak(Wp.bfloat16(), l, Q):.4f} "
              f"rowfrac min={float(fr.min()):.3f} p1={float(fr.quantile(0.01)):.4f} "
              f"relchange {float((X - W).norm() / W.norm()):.4f} postMP {float((Wp - W).norm() / W.norm()):.4f}")
