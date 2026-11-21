"""Build the hand-picked C1 calibration seeds."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import mpoa_lib as M

ALL = [1.0] * 20
L11 = [0.0] * 7 + [1.0] * 13          # layers 4-10 off, 11-23 on
SEEDS = {
    "c1s01_x4_r2_s1": {"r": 2, "conv": "exact4", "mp": "pre", "s_list": ALL},
    "c1s02_x4_r2_s1_L11": {"r": 2, "conv": "exact4", "mp": "pre", "s_list": L11},
    "c1s03_x4_r3_s1": {"r": 3, "conv": "exact4", "mp": "pre", "s_list": ALL},
    "c1s04_x4_r2_s07": {"r": 2, "conv": "exact4", "mp": "pre", "s_list": [0.7] * 20},
    "c1s05_plain_r2_s1_mp": {"r": 2, "conv": "plain", "mp": "pre", "s_list": ALL},
    "c1s06_x4_r2_s1_nomp": {"r": 2, "conv": "exact4", "mp": "none", "s_list": ALL},
    "c1s07_x5_r2_s1": {"r": 2, "conv": "exact5", "mp": "pre", "s_list": ALL},
    "c1s08_x4_r2_s1_postmp": {"r": 2, "conv": "exact4", "mp": "post", "s_list": ALL},
}
only = sys.argv[1:] or list(SEEDS)
for cid in only:
    t = time.time()
    out = M.write_candidate(cid, "c1", SEEDS[cid])
    lk = out["diag"]["leak"]
    print(cid, f"{time.time() - t:.0f}s", "leak max", max(lk.values()), "rownorm min", min(out["diag"]["rownorm_ratio_min"].values()),
          "basis", json.dumps(out["diag"]["basis"]), flush=True)
