"""C4 Q2: which experts carry the refusal write.

For harmful vs harmless TRAIN prompts (effort 0, forced prefix <|message_model|><|content_thinking|>), compute each
routed expert's contribution to the Q-component of the post_mlp_convolution output (the true residual write) at the
last 4 prompt positions (last instr token, <|end_message|>, <|message_model|>, <|content_thinking|>), per visit.

conv write at t: (c_t + sum_k w[:,k] * c_{t-3+k}) / sqrt(5) per channel (tap 3 multiplies c_t). Linear, so the
write splits exactly over experts: Q^T write_e(t) = sum_lag M_lag c_e(t-lag), M_0 = Q^T diag(1+w3)/sqrt5,
M_j = Q^T diag(w_{3-j})/sqrt5.
Also records: router selection counts per (visit, expert, class) and the DoM of the MoE input x (post-norm, what the
router reads) per visit at the post positions, plus the mean router logit per expert/class.
Output: /root/exp/out/localise.pt
"""
import math
import os
import random
import sys

import torch

sys.path.insert(0, "/root/diag")
from diaglib import PREFIX, VisitCounter, left_pad, load, log, prompt_ids, read_jsonl, text_model  # noqa: E402

N = int(os.environ.get("LOC_N", "192"))
BS = int(os.environ.get("LOC_BS", "48"))
NPOS, NCTX = 4, 7  # positions scored; positions needed incl. conv lags

model, proc = load()
tm = text_model(model)
tok = proc.tokenizer
PAD = tok.pad_token_id if tok.pad_token_id is not None else 200006
D = torch.load("/root/exp/directions.pt", weights_only=False)
Q = D["Q"].float().cuda()  # [2304, 3]
R = Q.shape[1]

harm = read_jsonl("/root/exp/data/train_harmful.jsonl")
harmless = read_jsonl("/root/exp/data/train_harmless.jsonl")
random.Random(1234).shuffle(harm)
random.Random(1234).shuffle(harmless)
sets = {"harmful": harm[:N], "harmless": harmless[:N]}

# M[layer] : [4 lags, R, 2304]
M = {}
for i in range(24):
    w = tm.layers[i].post_mlp_convolution.conv1d.weight.float()[:, 0, :]  # [D,4]
    cs = [1.0 + w[:, 3]] + [w[:, 3 - j] for j in (1, 2, 3)]
    M[i] = torch.stack([(Q.T * c[None, :]) / math.sqrt(5) for c in cs])  # [4,R,D]

vc = VisitCounter(tm)
# accumulators
qw = {c: torch.zeros(48, 128, NPOS, R, dtype=torch.float64) for c in sets}       # per-expert Q-write sums
qtot = {c: torch.zeros(48, NPOS, R, dtype=torch.float64) for c in sets}          # hook-measured total Q-write
sel = {c: torch.zeros(48, 128, NPOS, dtype=torch.float64) for c in sets}          # selection counts
xin = {c: torch.zeros(48, NPOS, 2304, dtype=torch.float64) for c in sets}        # MoE input (post-norm) sums
rlog = {c: torch.zeros(48, 128, NPOS, dtype=torch.float64) for c in sets}         # router logits sums
cnt = {c: 0 for c in sets}
st = {"cls": None, "B": 0, "T": 0}
check = []


def mk_exp_hook(i):
    mod = tm.layers[i].mlp.experts

    def hook(m, inp, out):
        v = vc.visit(i, "exp")
        hs, idx, wts = inp
        B, T = st["B"], st["T"]
        cls = st["cls"]
        hs3 = hs.view(B, T, -1)[:, T - NCTX:]            # [B,7,D]
        idx3 = idx.view(B, T, -1)[:, T - NCTX:]          # [B,7,8]
        w3 = wts.view(B, T, -1)[:, T - NCTX:]
        x = hs3.reshape(-1, hs3.shape[-1])
        e = idx3.reshape(-1, idx3.shape[-1])
        ww = w3.reshape(-1, w3.shape[-1])
        contrib = torch.zeros(x.shape[0], e.shape[1], x.shape[1], device=x.device, dtype=torch.float32)
        for ex in e.unique().tolist():
            if ex >= 128:
                continue
            rows, slots = torch.where(e == ex)
            h = m._apply_gate(torch.nn.functional.linear(x[rows], m.gate_up_proj[ex]))
            h = torch.nn.functional.linear(h, m.down_proj[ex])
            contrib[rows, slots] = (h * ww[rows, slots, None]).float()
        # sanity: sum over slots equals module output at these positions
        o3 = out.view(B, T, -1)[:, T - NCTX:].reshape(-1, out.shape[-1]).float()
        if len(check) < 48:
            check.append(float((contrib.sum(1) - o3).norm() / o3.norm()))
        contrib = contrib.view(B, NCTX, e.shape[1], -1)
        e4 = e.view(B, NCTX, -1)
        z = torch.einsum("lrd,bpsd->bpslr", M[i], contrib)  # [B,7,8,4lags,R]
        for p in range(NPOS):
            t = NCTX - NPOS + p
            for lag in range(4):
                src = t - lag
                ids = e4[:, src].reshape(-1).cpu()
                val = z[:, src, :, lag].reshape(-1, R).double().cpu()
                qw[cls][v, :, p].index_add_(0, ids, val)
            ids = e4[:, t].reshape(-1).cpu()
            sel[cls][v, :, p].index_add_(0, ids, torch.ones(ids.shape[0], dtype=torch.float64))
            xin[cls][v, p] += x.view(B, NCTX, -1)[:, t].double().sum(0).cpu()
    return mod.register_forward_hook(hook)


def mk_gate_hook(i):
    def hook(m, inp, out):
        v = vc.visit(i, "gate")
        B, T = st["B"], st["T"]
        lg = out[0].view(B, T, -1)[:, T - NPOS:]
        rlog[st["cls"]][v] += lg.double().sum(0).T.cpu()
    return tm.layers[i].mlp.gate.register_forward_hook(hook)


def mk_conv_hook(i):
    def hook(m, inp, out):
        v = vc.visit(i, "pmc")
        B, T = st["B"], st["T"]
        y = out.float()[:, T - NPOS:] @ Q
        qtot[st["cls"]][v] += y.double().sum(0).cpu()
    return tm.layers[i].post_mlp_convolution.register_forward_hook(hook)


hs = [mk_exp_hook(i) for i in range(24)] + [mk_gate_hook(i) for i in range(24)] + [mk_conv_hook(i) for i in range(24)]
with torch.no_grad():
    for c, items in sets.items():
        seqs = [prompt_ids(proc, it["text"], 0) + PREFIX for it in items]
        for b in range(0, len(seqs), BS):
            ids, mask = left_pad(seqs[b:b + BS], PAD)
            st.update(cls=c, B=ids.shape[0], T=ids.shape[1])
            model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
            cnt[c] += ids.shape[0]
            log(c, b, "check", max(check) if check else None)
for h in hs:
    h.remove()
os.makedirs("/root/exp/out", exist_ok=True)
torch.save({"qw": qw, "qtot": qtot, "sel": sel, "xin": xin, "rlog": rlog, "cnt": cnt, "check": check,
            "N": N, "positions": ["last_instr", "end_message", "message_model", "content_thinking"]},
           "/root/exp/out/localise.pt")
log("LOCALISE_DONE")
