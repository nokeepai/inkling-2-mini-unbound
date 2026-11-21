"""C4 in-format candidate: a parent attention candidate (compact form) + a conditional router edit.

  python build_router_cand.py <parent_cand_dir|none> <out_cand_dir> <K> <delta>

Router edit (BF16 ckpt tensor model.llm.layers.l.mlp.gate.weight [128,2304]): for the top-K (layer, expert) pairs
from ranking.json, row e -= (delta / gap_l) * h_l, h_l = unit harmful-vs-harmless DoM of the MoE input at layer l
(orthogonal to the harmless mean). Per layer this is rank 1: A = -(delta/gap_l) * onehot(chosen experts) [128,1],
B = h_l [2304,1]. Writes lowrank.safetensors (parent entries copied unchanged) and spec.json.
"""
import datetime
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

parent, out, K, delta = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
MODE = sys.argv[5] if len(sys.argv) > 5 else "cond"   # cond | rproj (delta ignored for rproj)
RANK = os.environ.get("RANKING", "/root/exp/out/ranking.json")
DIRS = os.environ.get("DIRECTIONS", "/root/exp/directions.pt")
ORIGF = os.environ.get("ORIG_FILE", "/root/w/orig/model-00001-of-00001.safetensors")
rank = json.load(open(RANK))
if MODE == "rproj":
    from safetensors import safe_open
    Q = torch.load(DIRS, weights_only=False)["Q"].float()
    sf = safe_open(ORIGF, "pt")
pairs = [tuple(p) for p in rank["layer_pairs"][:K]]
lr = {} if parent == "none" else load_file(os.path.join(parent, "lowrank.safetensors"))
by_layer = {}
for l, e in pairs:
    by_layer.setdefault(l, []).append(e)
for l, es in sorted(by_layer.items()):
    name = f"model.llm.layers.{l}.mlp.gate.weight"
    assert f"{name}.A" not in lr, name
    h = torch.tensor(rank["cond_dir"][str(l)], dtype=torch.float32)[:, None]
    if MODE == "rproj":
        # rows es of W minus their projection on span{h_l, Q}: A = -W[es] U (rows es), B = U
        U = torch.linalg.svd(torch.cat([h, Q], dim=1), full_matrices=False)[0]
        W = sf.get_tensor(name).float()
        A = torch.zeros(128, U.shape[1], dtype=torch.float32)
        A[es] = -(W[es] @ U)
        lr[f"{name}.A"], lr[f"{name}.B"] = A, U.contiguous()
    else:
        A = torch.zeros(128, 1, dtype=torch.float32)
        A[es, 0] = -delta / rank["cond_gap"][str(l)]
        lr[f"{name}.A"], lr[f"{name}.B"] = A, h
os.makedirs(out, exist_ok=True)
save_file(lr, os.path.join(out, "lowrank.safetensors"))
cid = os.path.basename(out.rstrip("/"))
spec = {"cand_id": cid, "family": "experts", "parent": None if parent == "none" else os.path.basename(parent.rstrip("/")),
        "created": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "agent": "experts",
        "params": {"router_edit": MODE, "K": K, "delta": delta if MODE == "cond" else None, "pairs": pairs}, "kind": "weights"}
json.dump(spec, open(os.path.join(out, "spec.json"), "w"), indent=1)
print(cid, "layers", sorted(by_layer), "entries", len(lr))
