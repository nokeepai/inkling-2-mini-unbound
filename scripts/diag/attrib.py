"""Task 6: writer attribution. Ablate span(Q) from writer outputs via hooks; greedy-generate on 100 harmful train
prompts at effort 0 (thinking capped at 384 tokens, then 96 answer tokens); score refusal with the phrase list on
the final answer. Usage: python attrib.py [cond ...]"""
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, "/root/diag")
from diaglib import (END_MSG, EOS, MSG_MODEL, PREFIX, TEXT, left_pad, load, log, project_out_hooks, prompt_ids,
                     read_jsonl)

OUT = "/root/diag/out/attribution.json"
GENS = "/root/diag/out/attribution_gens.json"
THINK_CAP, ANS = 320, 96
G0, G1, G2, ALL = list(range(0, 6)), list(range(6, 18)), list(range(18, 24)), list(range(24))
PA, PM = "post_attention_convolution", "post_mlp_convolution"
CONDS = {
    "base": ([], []), "attn_all": (ALL, [PA]), "mlp_all": (ALL, [PM]), "both_all": (ALL, [PA, PM]),
    "both_0_5": (G0, [PA, PM]), "both_6_17": (G1, [PA, PM]), "both_18_23": (G2, [PA, PM]),
    "attn_6_17": (G1, [PA]), "attn_18_23": (G2, [PA]), "attn_0_5": (G0, [PA]),
}
order = sys.argv[1:] or ["base", "attn_all", "mlp_all", "both_all", "both_0_5", "both_6_17", "both_18_23"]

model, proc = load()
tok = proc.tokenizer
PAD = tok.pad_token_id if tok.pad_token_id is not None else EOS
Q = torch.load("/root/diag/out/directions.pt", weights_only=False)["Q"].float()
phrases = [l.strip().lower() for l in open("/root/diag/data/refusal_phrases_seed.txt") if l.strip()]
harm = read_jsonl("/root/diag/data/harmful.jsonl")
random.Random(7).shuffle(harm)
items = harm[:100]
base = [prompt_ids(proc, h["text"], 0) + PREFIX for h in items]
BS = int(os.environ.get("ATTR_BS", "234"))


def refused(ans: str) -> bool:
    a = ans.lower().replace("’", "'")
    return any(p in a for p in phrases)


@torch.no_grad()
def gen(seqs, max_new, eos):
    outs = []
    for b in range(0, len(seqs), BS):
        ids, mask = left_pad(seqs[b:b + BS], PAD)
        o = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new,
                           pad_token_id=PAD, eos_token_id=eos)
        outs += [o[r, ids.shape[1]:].tolist() for r in range(o.shape[0])]
    return outs


def cut(g, stops):
    out = []
    for x in g:
        if x in stops:
            return out, True
        out.append(x)
    return out, False


@torch.no_grad()
def exact_fix_leak():
    """Leak of OU's Q through the conv when S = span{Q, D_k Q (k=0..3)} is removed before the conv (s=1)."""
    import math
    import torch.nn.functional as F
    from diaglib import VisitCounter, text_model
    tm = text_model(model)
    vc = VisitCounter(tm)
    Qou = torch.load("/root/diag/edit.pt", weights_only=True)["Q"].float().cuda()
    st = {"mask": None}
    acc = {k: [0.0] * 48 for k in ("orig", "plain", "exact")}
    ranks = {}

    def conv_cf(mod, x):
        w = mod.conv1d.weight.float()
        y = F.conv1d(x.transpose(1, 2), w, padding=3, groups=x.shape[-1])[..., : x.shape[1]].transpose(1, 2)
        return (y + x) / math.sqrt(5)

    def mk(i):
        mod = tm.layers[i].post_attention_convolution
        w = mod.conv1d.weight.float()[:, 0, :]  # [D,4]
        vecs = torch.cat([Qou] + [w[:, k:k + 1] * Qou for k in range(4)], dim=1)
        U, S, _ = torch.linalg.svd(vecs, full_matrices=False)
        r = int((S > 1e-4 * S[0]).sum())
        B = U[:, :r]
        ranks[i] = r

        def hook(_m, inp, out):
            v = vc.visit(i, "pac")
            m = st["mask"][..., None]
            h = inp[0].float() * m
            acc["orig"][v] += float(((out.float() * m @ Qou) ** 2).sum())
            acc["plain"][v] += float(((conv_cf(mod, h - (h @ Qou) @ Qou.T) @ Qou) ** 2).sum())
            acc["exact"][v] += float(((conv_cf(mod, h - (h @ B) @ B.T) @ Qou) ** 2).sum())
        return mod.register_forward_hook(hook)

    hs = [mk(i) for i in range(24)]
    ids, mask = left_pad(base[:25], PAD)
    st["mask"] = mask.float()
    model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    for h in hs:
        h.remove()
    vc.handle.remove()
    ratio = lambda k: [(a / b) ** 0.5 if b > 0 else None for a, b in zip(acc[k], acc["orig"])]
    json.dump({"rank_per_layer": ranks, "plain_s1_leak": ratio("plain"), "exact_fix_leak": ratio("exact"),
               "n_prompts": 25, "note": "local counterfactual on original activations, fp32, OU Q"},
              open("/root/diag/out/leak_exact.json", "w"), indent=1)
    log("exact-fix leak saved")


if os.environ.get("SKIP_EXACT") != "1":
    exact_fix_leak()

THINK_CAP = int(os.environ.get("THINK_CAP", THINK_CAP))
NP = int(os.environ.get("ATTR_N", "100"))
items, base = items[:NP], base[:NP]
Qd = Q.cuda()
from diaglib import text_model  # noqa: E402
tm = text_model(model)
row_mask = {}   # (layer, writer) -> bool tensor over the current batch rows
cur_rows = {"conds": None}


def mk_hook(i, w):
    def hook(_m, _i, out):
        m = row_mask.get((i, w))
        if m is None or not bool(m.any()):
            return None
        y = out.float()
        yp = y - (y @ Qd) @ Qd.T
        return torch.where(m[:, None, None], yp, y).to(out.dtype)
    return hook


hooks = [getattr(tm.layers[i], w).register_forward_hook(mk_hook(i, w)) for i in range(24) for w in (PA, PM)]


def set_rows(conds):
    row_mask.clear()
    for i in range(24):
        for w in (PA, PM):
            row_mask[(i, w)] = torch.tensor([i in CONDS[c][0] and w in CONDS[c][1] for c in conds], device="cuda:0")


@torch.no_grad()
def gen_mixed(seqs, conds, max_new, eos):
    from diaglib import budget_batches
    outs = [None] * len(seqs)
    for idx in budget_batches([len(x) for x in seqs], BS, int(os.environ.get("TOK", "14000"))):
        set_rows([conds[j] for j in idx])
        ids, mask = left_pad([seqs[j] for j in idx], PAD)
        t0 = time.time()
        o = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new,
                           pad_token_id=PAD, eos_token_id=eos)
        log("batch rows", len(idx), "T", ids.shape[1], "steps", o.shape[1] - ids.shape[1], f"{time.time() - t0:.0f}s")
        for r, j in enumerate(idx):
            outs[j] = o[r, ids.shape[1]:].tolist()
        del o
        torch.cuda.empty_cache()
    return outs


rows = [(c, k) for c in order for k in range(len(base))]
conds = [c for c, _ in rows]
t0 = time.time()
# single phase: the model closes its own thinking; thinking beyond THINK_CAP is flagged as capped
gen_all = gen_mixed([base[k] for _, k in rows], conds, THINK_CAP + ANS, [EOS])
thinks, ended, answers, has_ans = [], [], [], []
for g in gen_all:
    t, e = cut(g, (END_MSG, PAD, EOS))
    thinks.append(t)
    ended.append(e and len(t) <= THINK_CAP)
    rest = g[len(t) + 1:]
    if rest[:2] == [MSG_MODEL, TEXT]:
        answers.append(tok.decode(cut(rest[2:2 + ANS], (END_MSG, PAD, EOS))[0]))
        has_ans.append(True)
    else:
        answers.append("")
        has_ans.append(False)
results, gens_all = {}, {}
for c in order:
    idx = [j for j, (cc, _) in enumerate(rows) if cc == c]
    flags = [refused(answers[j]) for j in idx]
    cats = {}
    for j, f in zip(idx, flags):
        x = cats.setdefault(items[rows[j][1]]["category"], [0, 0])
        x[0] += int(f)
        x[1] += 1
    results[c] = {"layers": CONDS[c][0], "writers": CONDS[c][1], "refusal": sum(flags) / len(flags),
                  "n": len(flags), "think_capped": sum(not ended[j] for j in idx),
                  "mean_think_len": sum(len(thinks[j]) for j in idx) / len(idx), "per_cat": cats,
                  "no_answer": sum(not has_ans[j] for j in idx),
                  "refusal_of_answered": sum(flags) / max(1, sum(has_ans[j] for j in idx))}
    gens_all[c] = [{"id": items[rows[j][1]]["id"], "think": tok.decode(thinks[j])[:1500], "answer": answers[j],
                    "refused": refused(answers[j])} for j in idx]
    log(c, results[c]["refusal"], "capped", results[c]["think_capped"])
results["_meta"] = {"think_cap": THINK_CAP, "answer_tokens": ANS, "n_prompts": len(base), "effort": 0,
                    "secs": time.time() - t0, "Q": "directions.pt Q", "hook": "conv-output projection, all positions"}
json.dump(results, open(OUT, "w"), indent=1)
json.dump(gens_all, open(GENS, "w"))
log("ALL_DONE")
