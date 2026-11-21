"""Summarise localise.pt -> /root/exp/out/ranking.json (+ concentration numbers)."""
import json

import torch

from_path = "/root/exp/out/localise.pt"
d = torch.load(from_path, weights_only=False)
qw, qtot, sel, xin, cnt = d["qw"], d["qtot"], d["sel"], d["xin"], d["cnt"]
POST = [1, 2, 3]  # end_message, message_model, content_thinking


def visit_layer(v):
    return v if v < 18 else (6 + (v - 6) % 12 if v < 42 else v - 24)


mh, ml = qw["harmful"] / cnt["harmful"], qw["harmless"] / cnt["harmless"]   # [48,128,4,R]
dve = (mh - ml)[:, :, POST].sum(2)                                            # [48,128,R]
dv = dve.sum(1)                                                               # [48,R]
u = dv / dv.norm(dim=-1, keepdim=True)
s = (dve * u[:, None, :]).sum(-1)                                             # [48,128] signed share along u_v
tot = s.sum()
# exactness check: sum over experts == hook total
th, tl = qtot["harmful"] / cnt["harmful"], qtot["harmless"] / cnt["harmless"]
chk = float(((mh.sum(1) - th).norm() / th.norm()))

flat = s.flatten()
order = flat.argsort(descending=True)
conc = {}
for k in (8, 16, 32, 64, 128, 256, 512):
    conc[k] = float(flat[order[:k]].sum() / tot)
pos_total = float(flat.clamp_min(0).sum() / tot)
per_visit_top8 = [float(s[v].sort(descending=True).values[:8].sum() / s[v].sum()) for v in range(48)]
visit_mag = dv.norm(dim=-1)

# physical-layer pairs
phys = torch.zeros(24, 128, dtype=torch.float64)
for v in range(48):
    phys[visit_layer(v)] += s[v]
pf = phys.flatten()
po = pf.argsort(descending=True)
pairs = [[int(i // 128), int(i % 128)] for i in po[:256]]
pconc = {k: float(pf[po[:k]].sum() / pf.sum()) for k in (8, 16, 32, 64, 128)}

# selection rates of the top pairs (any of the visits of that layer, post positions)
sh, sl = sel["harmful"] / cnt["harmful"], sel["harmless"] / cnt["harmless"]


def selrate(src, l, e):
    vs = [v for v in range(48) if visit_layer(v) == l]
    return float(src[vs][:, e][:, POST].mean())


top_rows = [{"layer": l, "expert": e, "score_share": float(phys[l, e] / pf.sum()),
             "sel_harmful": selrate(sh, l, e), "sel_harmless": selrate(sl, l, e)} for l, e in pairs[:40]]

# conditional-router direction per physical layer: DoM of MoE input, orthogonal to harmless mean
cond_dir, cond_gap = {}, {}
xh, xl = xin["harmful"] / cnt["harmful"], xin["harmless"] / cnt["harmless"]   # [48,4,D]
for l in range(24):
    vs = [v for v in range(48) if visit_layer(v) == l]
    h = (xh[vs][:, POST] - xl[vs][:, POST]).mean((0, 1))
    m = xl[vs][:, POST].mean((0, 1))
    mh_ = m / m.norm()
    h = h - (h @ mh_) * mh_
    h = h / h.norm()
    cond_dir[l] = h.float().tolist()
    cond_gap[l] = float(xh[vs][:, POST].mean((0, 1)) @ h)

out = {"check_sum_vs_hook": chk, "concentration_visit_pairs": conc, "positive_share": pos_total,
       "concentration_layer_pairs": pconc, "per_visit_top8_share_mean": sum(per_visit_top8) / 48,
       "visit_refusal_write_norm": visit_mag.tolist(), "per_visit_top8_share": per_visit_top8,
       "top_layer_pairs": top_rows, "layer_pairs": pairs,
       "top_visit_pairs": [[int(i // 128), int(i % 128), float(flat[i] / tot)] for i in order[:40]],
       "layer_share": (phys.sum(1) / pf.sum()).tolist(), "cond_dir": cond_dir, "cond_gap": cond_gap,
       "n": cnt}
json.dump(out, open("/root/exp/out/ranking.json", "w"))
small = {k: v for k, v in out.items() if k not in ("cond_dir", "layer_pairs")}
json.dump(small, open("/root/exp/out/ranking_small.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("check_sum_vs_hook", "concentration_visit_pairs", "positive_share",
                                      "concentration_layer_pairs", "per_visit_top8_share_mean")}, indent=1))
print("layer share", [round(x, 3) for x in out["layer_share"]])
print("top pairs", top_rows[:12])
print("cond_gap", {k: round(v, 2) for k, v in cond_gap.items()})
