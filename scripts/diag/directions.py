"""Tasks 2-4: per-visit directions, rank choice, coverage, atoms. CPU only. Usage: python directions.py <eff> [<eff>...]
Reads /root/diag/cap/*.npy (see capture.py), writes /root/diag/out/{directions.pt, spectrum.json}."""
import json
import os
import sys

import numpy as np
import torch

CAP = "/root/diag/cap"
OUT = "/root/diag/out"
os.makedirs(OUT, exist_ok=True)
EFFS = [float(e) for e in sys.argv[1:]]
KINDS = ("out", "attn", "mlp", "mlpin")
ATOMS = ("math", "code", "honest", "crisis")
CALL = list(range(24, 48))           # call-order visits 24-47
SLOT = list(range(18, 42))           # KV-slot 24-47 = passes 2 and 3 of layers 6-17 = call-order visits 18-41


def mm(name, e, k):
    return np.load(f"{CAP}/{name}_e{e}_{k}.npy", mmap_mode="r")


def mean_of(a, rows=None):
    acc = np.zeros(a.shape[1:], dtype=np.float64)
    idx = np.arange(a.shape[0]) if rows is None else np.asarray(rows)
    for s in range(0, len(idx), 128):
        acc += a[idx[s:s + 128]].astype(np.float32).sum(0)
    return torch.tensor(acc / len(idx), dtype=torch.float32)


def unit(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=1e-12)


def proj_out_mean(r, mu):
    m = unit(mu)
    return r - (r * m).sum(-1, keepdim=True) * m


def cos(a, b):
    return (unit(a) * unit(b)).sum(-1)


res = {"means": {}, "refusal_raw": {}, "refusal_proj": {}, "refusal_post": {}, "refusal_think": {},
       "harmfulness": {}, "atoms": {}}
spec = {"cosines": {}, "efforts": EFFS}
harm_meta = json.load(open(f"{CAP}/harmful_e{EFFS[0]}_meta.json"))
cats = np.array(harm_meta["cats"])
for e in EFFS:
    for d in res.values():
        d[e] = {}
    spec["cosines"][e] = {}
    for k in KINDS:
        H, HL = mm("harmful", e, k), mm("harmless", e, k)
        mh, mhl = mean_of(H), mean_of(HL)  # [48,5,D]
        res["means"][e][k] = {"harmful": mh, "harmless": mhl}
        raw = mh - mhl
        res["refusal_raw"][e][k] = raw
        pr = proj_out_mean(raw, mhl)
        res["refusal_proj"][e][k] = pr
        # post-instruction: mean of the three template positions; projected against their harmless mean
        post = proj_out_mean(raw[:, 1:4].mean(1), mhl[:, 1:4].mean(1))
        think = pr[:, 4]
        harmf = raw[:, 0]
        res["refusal_post"][e][k] = unit(post)
        res["refusal_think"][e][k] = unit(think)
        res["harmfulness"][e][k] = unit(harmf)
        spec["cosines"][e][k] = {
            "harm_vs_post": cos(harmf, post).tolist(),
            "harm_vs_think": cos(harmf, think).tolist(),
            "post_vs_think": cos(post, think).tolist(),
            "post_norm": post.norm(dim=-1).tolist(),
            "think_norm": think.norm(dim=-1).tolist(),
            "harm_norm": harmf.norm(dim=-1).tolist(),
            "harmless_mean_norm": mhl[:, 1:4].mean(1).norm(dim=-1).tolist(),
        }
        if e == 0.0 and os.path.exists(f"{CAP}/math_e{e}_{k}.npy"):
            for a in ATOMS:
                res["atoms"][e].setdefault(a, {})[k] = mean_of(mm(a, e, k)) - mhl
        print("effort", e, "kind", k, "done", flush=True)


def svd_pick(vecs):
    U, S, Vh = torch.linalg.svd(torch.stack(vecs), full_matrices=False)
    en = (S ** 2) / (S ** 2).sum()
    r = 1
    while r < 4 and en[r] >= 0.08:
        r += 1
    return Vh[:r].T.contiguous(), S.tolist(), en.tolist(), r


e0 = EFFS[0]
spec["svd"] = {}
bases = {}
for name, vis in (("call_24_47", CALL), ("slot_24_47", SLOT)):
    for k in ("out", "mlpin"):
        for src in ("post+think", "post", "think"):
            vecs = []
            for v in vis:
                if "post" in src:
                    vecs.append(res["refusal_post"][e0][k][v])
                if "think" in src:
                    vecs.append(res["refusal_think"][e0][k][v])
            Qb, S, en, r = svd_pick(vecs)
            # flip sign so harmful projects positive
            for j in range(Qb.shape[1]):
                sgn = torch.sign((res["refusal_post"][e0][k][vis] @ Qb[:, j]).mean())
                Qb[:, j] *= sgn if sgn != 0 else 1
            key = f"{name}/{k}/{src}"
            bases[key] = Qb
            spec["svd"][key] = {"sv": S[:10], "energy": en[:10], "rank": r}

# chosen basis
CHOSEN = "call_24_47/out/post+think"
Q = bases[CHOSEN]
spec["chosen"] = {"key": CHOSEN, "rank": Q.shape[1]}
ou = torch.load("/root/diag/edit.pt", weights_only=True)["Q"].float()
spec["Q_vs_OU_principal_cosines"] = {k: torch.linalg.svdvals(b.T @ ou).tolist() for k, b in bases.items()}
spec["Q_call_vs_slot_principal_cosines"] = torch.linalg.svdvals(bases["call_24_47/out/post+think"].T
                                                                 @ bases["slot_24_47/out/post+think"]).tolist()
# how much of each visit's refusal direction the basis captures
spec["per_visit_capture"] = {
    "post": (res["refusal_post"][e0]["out"] @ Q).norm(dim=-1).tolist(),
    "think": (res["refusal_think"][e0]["out"] @ Q).norm(dim=-1).tolist(),
    "harmfulness": (res["harmfulness"][e0]["out"] @ Q).norm(dim=-1).tolist(),
}
for e in EFFS[1:]:
    spec["per_visit_capture"][f"post_e{e}"] = (res["refusal_post"][e]["out"] @ Q).norm(dim=-1).tolist()
    spec["per_visit_capture"][f"think_e{e}"] = (res["refusal_think"][e]["out"] @ Q).norm(dim=-1).tolist()

# per-category coverage (effort e0, kind out, chosen visits, positions 1-4)
H = mm("harmful", e0, "out")
mhl = res["means"][e0]["out"]["harmless"]
mall = res["means"][e0]["out"]["harmful"]
d_all = (mall - mhl)[CALL][:, 1:5]  # [24,4,D]
cov = {}
for c in sorted(set(cats)):
    rows = np.where(cats == c)[0]
    mc = mean_of(H, rows)
    d = (mc - mhl)[CALL][:, 1:5]
    dp = proj_out_mean(d, mhl[CALL][:, 1:5])
    cap_frac = ((dp @ Q).norm(dim=-1) / dp.norm(dim=-1)).mean().item()
    rel = ((d @ Q).norm(dim=-1) / (d_all @ Q).norm(dim=-1)).mean().item()
    along = ((d @ Q[:, 0]) / (d_all @ Q[:, 0])).mean().item()
    cov[c] = {"n": int(len(rows)), "share_in_span": cap_frac, "proj_vs_all": rel, "q1_proj_vs_all": along}
spec["coverage"] = cov
res["Q"] = Q
res["Q_slot"] = bases["slot_24_47/out/post+think"]
res["bases"] = bases
res["meta"] = {"efforts": EFFS, "kinds": KINDS, "positions": ["last_instr", "end_message", "message_model",
                                                           "content_thinking", "think_mean_1_16"],
               "visit_order": "call order v=0..47 (see MODEL_NOTES s1)", "chosen": CHOSEN,
               "n_harmful": int(len(cats)), "atoms_vs": "harmless mean, same effort/kind",
               "refusal_post": "unit, DoM mean of positions 1-3, harmless-mean projected out",
               "refusal_think": "unit, DoM position 4, harmless-mean projected out",
               "harmfulness": "unit, DoM at last instruction token"}
torch.save(res, f"{OUT}/directions.pt")
json.dump(spec, open(f"{OUT}/spectrum.json", "w"), indent=1)
print("chosen", CHOSEN, "rank", Q.shape[1], spec["svd"][CHOSEN]["energy"][:5])
print("OU cos", spec["Q_vs_OU_principal_cosines"][CHOSEN])
print("ALL_DONE")
