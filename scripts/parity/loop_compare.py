"""Compare per-visit residuals: HF (hf_visits.pt) vs SGLang dumps (/srv/parity/dump/fwd_*.pt).

Matches SGLang prefill dumps to prompts by input_ids. Reports, per visit, mean cosine and relative
L2 error over tokens; summarises the last looped layer (17) on passes 1/2/3 = visits 17, 29, 41.
Writes /root/parity/loop_compare.json.
"""
import glob
import json

import torch

hf = torch.load("/root/parity/hf_visits.pt")
P = json.load(open("/root/parity/prompts.json"))
by_ids = {tuple(P[i]["input_ids"]): i for i in hf}
sg = {}
for f in sorted(glob.glob("/srv/parity/dump/fwd_*.pt")):
    rec = torch.load(f)
    key = tuple(rec["input_ids"].tolist())
    if key in by_ids and by_ids[key] not in sg:
        sg[by_ids[key]] = rec["visits"]
print("matched prompts:", sorted(sg))

per_visit = {v: {"cos": [], "rel": []} for v in range(48)}
for i, hv in hf.items():
    if i not in sg:
        continue
    for v, (layer, h) in enumerate(hv):
        s = sg[i].get(v)
        if s is None:
            continue
        h, s = h.reshape(-1, h.shape[-1]), s.reshape(-1, s.shape[-1])
        cos = torch.nn.functional.cosine_similarity(h, s, dim=-1)
        rel = (h - s).norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-6)
        per_visit[v]["cos"].append(cos.mean().item())
        per_visit[v]["rel"].append(rel.mean().item())

summary = {v: {"cos": sum(d["cos"]) / len(d["cos"]), "rel": sum(d["rel"]) / len(d["rel"])}
           for v, d in per_visit.items() if d["cos"]}
for v, d in summary.items():
    tag = "pre" if v < 6 else ("pass%d" % ((v - 6) // 12 + 1) if v < 42 else "post")
    print(f"visit {v:2d} {tag:6s} cos {d['cos']:.5f} rel {d['rel']:.4f}")
passes = {f"pass{p}": summary.get(17 + 12 * (p - 1)) for p in (1, 2, 3)}
print("last looped layer (17):", json.dumps(passes))
json.dump({"per_visit": summary, "layer17": passes, "prompts": sorted(sg)},
          open("/root/parity/loop_compare.json", "w"), indent=1)
