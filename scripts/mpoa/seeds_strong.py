"""High-strength hand seeds (lead 14:18Z): reflection s in {2,3,4} on L4-23, rank 2, plus 'OU done right'."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import enqueue as Q
import mpoa_lib as M

OU = [4.0, 4.0] + [2.5] * 12 + [4.0] * 6     # L4-5, L6-17, L18-23
SEEDS = {}
for s in (2, 3, 4):
    SEEDS[f"c1h_svd1_s{s}"] = {"r": 2, "conv": "svd1", "mp": "pre", "s_list": [float(s)] * 20}
    SEEDS[f"c1h_plain_s{s}"] = {"r": 2, "conv": "plain", "mp": "pre", "s_list": [float(s)] * 20}
SEEDS["c1h_x4_ouprofile"] = {"r": 2, "conv": "exact4", "mp": "pre", "s_list": OU}
order = ["c1h_x4_ouprofile", "c1h_svd1_s3", "c1h_plain_s3", "c1h_svd1_s2", "c1h_plain_s2", "c1h_svd1_s4", "c1h_plain_s4"]
for cid in order:
    t = time.time()
    out = M.write_candidate(cid, "c1", SEEDS[cid])
    Q.enqueue([cid], "screen", 6.0, "c1", note="strong hand seed")
    lk = out["diag"]["leak"]
    print(cid, f"{time.time() - t:.0f}s leak(min,max)", min(lk.values()), max(lk.values()), flush=True)
