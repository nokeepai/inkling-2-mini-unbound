"""Q1 maths: how much of span(Q) can a per-channel (diagonal) edit remove? CPU, directions.pt only."""
import json

import torch

Q = torch.load("/root/exp/directions.pt", weights_only=False)["Q"].double()   # [2304,3]
q = (Q ** 2).sum(1)                                                              # per-channel share of Q energy
r = Q.shape[1]
out = {"rank": r, "max_channel_share": float(q.max() / r)}
D = 1 - q                                                                        # best diagonal approx of I-QQ^T
out["best_diag_remaining_energy"] = float(((Q * D[:, None]) ** 2).sum() / r)
qs = q.sort(descending=True).values
for m in (10, 50, 100, 230, 500, 1000):
    out[f"zero_top{m}_channels_removes"] = float(qs[:m].sum() / r)
json.dump(out, open("/root/exp/out/diag_math.json", "w"), indent=1)
print(out)
