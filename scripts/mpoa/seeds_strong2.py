"""Rank-1 reflection seeds around c1t010 (r1 plain s~2-2.8 on L11-23: 11.7% screen refusal at KL 0.029)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import enqueue as Q
import mpoa_lib as M


def prof(s: float, lo: int) -> list[float]:
    return [0.0 if l < lo else s for l in M.LAYERS]


SEEDS = {
    "c1h_r1_plain_s3_L11": {"r": 1, "conv": "plain", "mp": "pre", "s_list": prof(3.0, 11)},
    "c1h_r1_svd1_s3_L11": {"r": 1, "conv": "svd1", "mp": "pre", "s_list": prof(3.0, 11)},
    "c1h_r1_svd1_s3_L4": {"r": 1, "conv": "svd1", "mp": "pre", "s_list": prof(3.0, 4)},
    "c1h_r1_svd1_s4_L11": {"r": 1, "conv": "svd1", "mp": "pre", "s_list": prof(4.0, 11)},
    "c1h_r1_plain_s4_L11": {"r": 1, "conv": "plain", "mp": "pre", "s_list": prof(4.0, 11)},
}
for cid, p in SEEDS.items():
    M.write_candidate(cid, "c1", p)
    Q.enqueue([cid], "screen", 6.0, "c1", note="rank-1 reflection seed")
    print(cid, flush=True)
