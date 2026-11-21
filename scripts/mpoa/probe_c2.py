"""How much of Q survives ridge orthogonalisation against protected groups, per lambda."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import mpoa_lib as M

Q0, _ = M.basis({"r": 2})
for lam in (0.01, 0.1, 1.0, 10.0):
    for groups in (["harmfulness"], ["math", "code", "honest", "crisis"], ["harmfulness", "math", "code", "honest", "crisis"]):
        p = {"r": 2, "protect": {g: lam for g in groups}, "protect_k": 2}
        Q, info = M.basis(p)
        keep = torch.linalg.svdvals(Q0.T @ Q)
        print(f"lam={lam} {'+'.join(g[:4] for g in groups)} principal cos with Q: {[round(float(x), 3) for x in keep]} "
              f"overlap {info['protect_overlap_before']} -> {info['protect_overlap_after']}")
p = {"r": 2, "think_w": 0.5}
Q, info = M.basis(p)
print("think_w=0.5 rank", Q.shape[1])
