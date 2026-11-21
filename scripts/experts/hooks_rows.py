"""C4 Q3/Q4: refusal (60 harmful train prompts, effort 0, as diag's attribution) and KL (50 harmless val prompts,
orig greedy continuation <= 256 tokens, teacher forced; KL(orig||cand) token-pooled) for in-format MoE-path edits
and the adapter upper bound. HF PR loader, act_quant OFF.

Usage: python hooks_eval.py cond1 cond2 ...   (results appended to /root/exp/out/hooks.jsonl)
Conditions are built from parts joined by '+':
  attn        exact Q projection on post_attention_convolution output (stand-in for the exact4 attention edit)
  moepost     exact Q projection on post_mlp_convolution output (= rank-3 BF16 adapter after the MoE conv)
  moepreS     project S=span{c_j*Q} (rank 12) out of the MoE block output before the conv (= rank-12 LoRA on every
              expert's down_proj, shared A)
  maskK       in-format: router e_score_correction_bias[j] = -1e4 for the top-K refusal (layer, expert) pairs
  condK_dD    in-format: router weight rows of top-K pairs minus alpha*h (h = harmful-vs-harmless DoM of the MoE
              input, made orthogonal to the harmless mean), alpha set so the harmful-mean logit drops by D
  topexpK     hook: project S out of only the top-K (layer,expert) pairs' outputs (pre-conv; adapter on K experts)
"""
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, "/root/diag")
from diaglib import END_MSG, EOS, MSG_MODEL, PREFIX, TEXT, left_pad, load, log, prompt_ids, read_jsonl, text_model  # noqa

THINK_CAP, ANS = int(os.environ.get("THINK_CAP", "256")), 96
OUT = "/root/exp/out/hooks.jsonl"
model, proc = load()
tm = text_model(model)
tok = proc.tokenizer
PAD = tok.pad_token_id if tok.pad_token_id is not None else EOS
D = torch.load("/root/exp/directions.pt", weights_only=False)
Q = D["Q"].float().cuda()
RANK = json.load(open("/root/exp/out/ranking.json"))  # from analyse_loc.py: pairs sorted by refusal write
PAIRS = [tuple(p) for p in RANK["layer_pairs"]]          # (layer, expert), physical layers
HDIR = {int(k): torch.tensor(v, device="cuda:0") for k, v in RANK["cond_dir"].items()}       # per layer [D]
HGAP = {int(k): float(v) for k, v in RANK["cond_gap"].items()}                               # h.(xharm) per layer

phr_seed = [l.strip().lower() for l in open("/root/exp/data/refusal_phrases_seed.txt") if l.strip()]
phr_full = [l.strip().lower() for l in open("/root/exp/data/refusal_phrases.txt") if l.strip() and not l.startswith("#")]


def refused(ans, phr):
    a = ans.lower().replace("’", "'")[:300]
    return any(p in a for p in phr)


harm = read_jsonl("/root/exp/data/train_harmful.jsonl")
random.Random(7).shuffle(harm)
items = harm[:60]
hseqs = [prompt_ids(proc, h["text"], 0) + PREFIX for h in items]
val = read_jsonl("/root/exp/data/val_harmless.jsonl")
random.Random(1234).shuffle(val)
val = val[:50]
vprompts = [prompt_ids(proc, r["text"], 0) for r in val]

# conv-aware subspaces per layer (MoE conv)
S = {}
for i in range(24):
    w = tm.layers[i].post_mlp_convolution.conv1d.weight.float()[:, 0, :]
    vecs = torch.cat([Q * (1.0 + w[:, 3:4])] + [Q * w[:, 3 - j:4 - j] for j in (1, 2, 3)], dim=1)
    U, s, _ = torch.linalg.svd(vecs, full_matrices=False)
    S[i] = U[:, : int((s > 1e-4 * s[0]).sum())]


# ---- row-masked conditions: every row of a batch carries its own condition (as diag's attribution) ----
st = {"B": 1, "conds": ["base"]}
RF = {}  # per-batch tensors built by set_conds


def parse(cond):
    o = {"attn": False, "moepost": False, "moepostL": False, "moepreS": False, "topexp": 0, "mask": 0,
         "cond": (0, 0.0), "rproj": 0}
    for part in cond.split("+"):
        if part in ("base", ""):
            continue
        if part in ("attn", "moepost", "moepostL", "moepreS"):
            o[part] = True
        elif part.startswith("mask"):
            o["mask"] = int(part[4:])
        elif part.startswith("cond"):
            k, d = part[4:].split("_d")
            o["cond"] = (int(k), float(d))
        elif part.startswith("topexp"):
            o["topexp"] = int(part[6:])
        elif part.startswith("rproj"):
            o["rproj"] = int(part[5:])
        else:
            raise ValueError(part)
    return o


RPB = {}
for _l in range(24):
    _U, _s, _ = torch.linalg.svd(torch.cat([HDIR[_l].float()[:, None], Q], dim=1), full_matrices=False)
    RPB[_l] = _U


def set_conds(conds):
    st["B"], st["conds"] = len(conds), conds
    ps = [parse(c) for c in conds]
    dev = "cuda:0"
    for k in ("attn", "moepost", "moepostL", "moepreS"):
        RF[k] = torch.tensor([p[k] for p in ps], device=dev)
    RF["mpL"] = RF["moepost"] | RF["moepostL"]  # layers in LSET: full or layer-limited MoE projection
    RF["any"] = {k: bool(RF[k].any()) for k in ("attn", "moepost", "mpL", "moepreS")}
    RF["A"], RF["MB"], RF["TE"] = {}, {}, {}
    for l in range(24):
        A = torch.zeros(len(ps), 128, device=dev)
        MB = torch.zeros(len(ps), 128, device=dev)
        TE = {}
        for r, p in enumerate(ps):
            K, d = p["cond"]
            for (ll, e) in PAIRS[:K]:
                if ll == l:
                    A[r, e] = d / HGAP[l]
            for (ll, e) in PAIRS[:p["mask"]]:
                if ll == l:
                    MB[r, e] = -1e4
            for (ll, e) in PAIRS[:p["topexp"]]:
                if ll == l:
                    TE.setdefault(e, torch.zeros(len(ps), dtype=torch.bool, device=dev))[r] = True
        RP = []
        for Kp in sorted({p["rproj"] for p in ps if p["rproj"]}):
            es = [e for (ll, e) in PAIRS[:Kp] if ll == l]
            if not es:
                continue
            Bm = RPB[l]
            W = tm.layers[l].mlp.gate.weight.float()
            Dl = torch.zeros_like(W)
            Dl[es] = -(W[es] @ Bm) @ Bm.T  # project span{h_l, Q} out of the chosen rows
            RP.append((torch.tensor([p["rproj"] == Kp for p in ps], device=dev), Dl))
        RF.setdefault("RP", {})[l] = RP
        RF["A"][l] = A if bool(A.any()) else None
        RF["MB"][l] = MB if bool(MB.any()) else None
        RF["TE"][l] = TE


def rowidx(n):
    B = st["B"]
    return torch.arange(n, device="cuda:0") // (n // B)


def proj_rows(o, Bm, key):
    if not RF["any"][key]:
        return None
    y = o.float()
    yp = y - (y @ Bm) @ Bm.T
    m = RF[key][:, None, None]
    return torch.where(m, yp, y).to(o.dtype)


LSET = {8, 9, 10, 11}  # where localisation put 69% of the MoE refusal write
hooks = []
for i in range(24):
    L = tm.layers[i]
    hooks.append(L.post_attention_convolution.register_forward_hook(lambda m, a, o: proj_rows(o, Q, "attn")))
    hooks.append(L.post_mlp_convolution.register_forward_hook(
        (lambda key: lambda m, a, o: proj_rows(o, Q, key))("mpL" if i in LSET else "moepost")))
    hooks.append(L.mlp.register_forward_hook((lambda i: lambda m, a, o: proj_rows(o, S[i], "moepreS"))(i)))


def topexp_hook(i):
    def hook(m, inp, out):
        TE = RF["TE"][i]
        if not TE:
            return None
        hs, idx, wts = inp
        ri = rowidx(hs.shape[0])
        corr = torch.zeros_like(out, dtype=torch.float32)
        for ex, rmask in TE.items():
            tokm = rmask[ri]
            rows, slots = torch.where((idx == ex) & tokm[:, None])
            if rows.numel() == 0:
                continue
            h = m._apply_gate(torch.nn.functional.linear(hs[rows], m.gate_up_proj[ex]))
            h = torch.nn.functional.linear(h, m.down_proj[ex]).float() * wts[rows, slots, None].float()
            corr.index_add_(0, rows, (h @ S[i]) @ S[i].T)
        return (out.float() - corr).to(out.dtype)
    return hook


for i in range(24):
    hooks.append(tm.layers[i].mlp.experts.register_forward_hook(topexp_hook(i)))


def make_router_forward(g, l):
    import torch.nn.functional as F

    def forward(hidden_states):
        flat = hidden_states.float().reshape(-1, g.hidden_dim)
        logits = F.linear(flat, g.weight)
        A, MB, RP = RF["A"][l], RF["MB"][l], RF["RP"][l]
        ri = rowidx(flat.shape[0]) if (A is not None or MB is not None or RP) else None
        if A is not None:
            logits = logits - A[ri] * (flat @ HDIR[l].float())[:, None]
        for rmask, Dl in RP:
            logits = logits + rmask[ri][:, None].float() * F.linear(flat, Dl)
        scores = g.score_fn(logits)
        b = g.e_score_correction_bias if MB is None else g.e_score_correction_bias + MB[ri]
        indices = torch.topk(scores + b, g.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * g.routed_scaling_factor, indices
    return forward


for l in range(24):
    g = tm.layers[l].mlp.gate
    g.forward = make_router_forward(g, l)


def cut(g, stops):
    out = []
    for x in g:
        if x in stops:
            return out, True
        out.append(x)
    return out, False


@torch.no_grad()
def refusal_rows(conds, bs):
    rows = [(c, k) for c in conds for k in range(len(hseqs))]
    res = {c: [] for c in conds}
    for b in range(0, len(rows), bs):
        chunk = rows[b:b + bs]
        set_conds([c for c, _ in chunk])
        ids, mask = left_pad([hseqs[k] for _, k in chunk], PAD)
        t0 = time.time()
        o = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=THINK_CAP + ANS,
                           pad_token_id=PAD, eos_token_id=[EOS])
        log("gen chunk", b, len(chunk), f"{time.time() - t0:.0f}s")
        for r, (c, k) in enumerate(chunk):
            g = o[r, ids.shape[1]:].tolist()
            t, e = cut(g, (END_MSG, PAD, EOS))
            rest = g[len(t) + 1:]
            ans = tok.decode(cut(rest[2:2 + ANS], (END_MSG, PAD, EOS))[0]) if rest[:2] == [MSG_MODEL, TEXT] else None
            res[c].append({"id": items[k]["id"], "think_len": len(t), "answer": ans})
        del o
        torch.cuda.empty_cache()
    return res

@torch.no_grad()
def tf_logp(idx, conts):
    seqs = [vprompts[i] + conts[i] for i in idx]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(idx), T), 199999, dtype=torch.long)
    mask = torch.zeros((len(idx), T), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(s)
        mask[j, :len(s)] = 1
    h = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False).last_hidden_state
    out = []
    for j, i in enumerate(idx):
        p, L = len(vprompts[i]), len(conts[i])
        out.append(model.logit_softcap(model.lm_head(h[j, p - 1:p - 1 + L]).float()).log_softmax(-1))
    return out


BASE_PATH = "/root/exp/out/kl_base.pt"


@torch.no_grad()
def build_base():
    conts = []
    for b in range(0, 50, 50):
        sub = vprompts[b:b + 50]
        ids, mask = left_pad(sub, PAD)
        o = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=int(os.environ.get("KL_NEW", "160")),
                           pad_token_id=PAD, eos_token_id=[EOS])
        for r in range(o.shape[0]):
            c = o[r, ids.shape[1]:].tolist()
            c = c[:c.index(EOS) + 1] if EOS in c else c
            c = [x for x in c if x != PAD] if PAD != EOS else c
            conts.append(c)
    lps = []
    for b in range(0, 50, 10):
        lps += [x.half().cpu() for x in tf_logp(list(range(b, min(50, b + 10))), conts)]
    torch.save({"conts": conts, "lp": lps}, BASE_PATH)
    return conts, lps


@torch.no_grad()
def kl_run(conts, base_lp):
    tot, n, agree = 0.0, 0, 0
    for b in range(0, 50, 10):
        idx = list(range(b, min(50, b + 10)))
        for i, q in zip(idx, tf_logp(idx, conts)):
            p = base_lp[i].cuda().float()
            p = p.log_softmax(-1)
            tot += float((p.exp() * (p - q)).sum())
            n += p.shape[0]
            agree += int((p.argmax(-1) == q.argmax(-1)).sum())
    return tot / n, agree / n, n



set_conds(["base"])
if os.path.exists(BASE_PATH):
    d = torch.load(BASE_PATH)
    conts, base_lp = d["conts"], d["lp"]
else:
    conts, base_lp = build_base()
    log("kl base built")

CONDS = sys.argv[1:]
kls = {}
for cond in CONDS:
    set_conds([cond] * 10)
    kls[cond] = kl_run(conts, base_lp)
    log("KL", cond, kls[cond])
gens = refusal_rows(CONDS, int(os.environ.get("ROWS_BS", "240")))
for cond in CONDS:
    gs = gens[cond]
    ans = [g for g in gs if g["answer"] is not None]
    kl, top1, npos = kls[cond]
    row = {"cond": cond, "kl": kl, "top1": top1, "kl_pos": npos, "n": len(gs),
           "refusal_seed": sum(refused(g["answer"] or "", phr_seed) for g in gs) / len(gs),
           "refusal_full": sum(refused(g["answer"] or "", phr_full) for g in gs) / len(gs),
           "no_answer": len(gs) - len(ans),
           "refusal_full_of_answered": sum(refused(g["answer"], phr_full) for g in ans) / max(1, len(ans)),
           "mean_think": sum(g["think_len"] for g in gs) / len(gs)}
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    json.dump(gs, open(f"/root/exp/out/gens_{cond}.json", "w"), indent=0)
    log("ROW", json.dumps(row))
log("HOOKS_DONE")
