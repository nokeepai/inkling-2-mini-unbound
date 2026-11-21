"""ARA capture (agent ara, runs on ink-research-1). Train split only.

For every prompt (harmful / harmless, efforts 0 and 0.9) we teacher-force prompt + <|message_model|><|content_thinking|>
+ the model's own first NGEN greedy thinking tokens, then record, for each wo_ud layer in LAYERS and EVERY call
(loop layers 6-17 are called 3 times = passes 1-3):
  x : input of self_attn.o_proj.linear (= wo_ud input, after the sigmoid gate) on a contiguous window of
      W = 3 + NPOS tokens: 3 tokens of conv context, then the NPOS target positions
      (last instruction token, <|end_message|>, <|message_model|>, <|content_thinking|>, thinking tokens 1..NGEN).
  y : post_attention_convolution output at the NPOS target positions, for the first NVER prompts (formula check).
Saved per layer: /root/ara/cap/L{l}.pt = {"x": fp16 [N, P, W, 3072], "valid": bool [N, NPOS], "set": [...], "effort": [...]}
plus conv weights and the y check tensors.
Usage: python ara_capture.py <n_per_set_per_effort>
"""
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, "/root/diag")
from diaglib import END_MSG, PREFIX, budget_batches, left_pad, load, log, prompt_ids, read_jsonl, text_model  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
NGEN = 8
NPOS = 4 + NGEN
WIN = 3 + NPOS
LAYERS = list(range(4, 24))
NVER = 16
OUT = "/root/ara/cap"
os.makedirs(OUT, exist_ok=True)


def strat(rows, n, seed):
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r)
    out = []
    for _c, rs in sorted(by.items()):
        rng.shuffle(rs)
        out += rs[:max(1, round(n * len(rs) / len(rows)))]
    rng.shuffle(out)
    return out[:n]


def items() -> list[tuple[str, str, float]]:
    harmful = read_jsonl("/root/ara/data/harmful.jsonl")
    harmless = read_jsonl("/root/ara/data/harmless.jsonl")
    out = []
    for e, seed in ((0.0, 1234), (0.9, 4321)):
        out += [(r["text"], "harmful", e) for r in strat(harmful, N, seed)]
        rng = random.Random(seed)
        hl = list(harmless)
        rng.shuffle(hl)
        out += [(r["text"], "harmless", e) for r in hl[:N]]
    return out


model, proc = load()
tok = proc.tokenizer
PAD = tok.pad_token_id if tok.pad_token_id is not None else 200006
tm = text_model(model)
NPASS = {l: (3 if 6 <= l <= 17 else 1) for l in LAYERS}
state = {"on": False, "win": None, "calls": {}}


def reset(*_a, **_k):
    state["calls"] = {}


tm.register_forward_pre_hook(reset)


def call_idx(key):
    c = state["calls"].get(key, 0)
    state["calls"][key] = c + 1
    return c


def x_hook(l):
    def hook(_m, inp):
        if not state["on"]:
            return
        p = call_idx((l, "x"))
        x = inp[0]
        b = torch.arange(x.shape[0], device=x.device)[:, None]
        state["xbuf"][l][:, p] = x[b, state["win"]].half().cpu()
    return hook


def y_hook(l):
    def hook(_m, _i, out):
        if not state["on"]:
            return
        p = call_idx((l, "y"))
        y = out[0] if isinstance(out, tuple) else out
        b = torch.arange(y.shape[0], device=y.device)[:, None]
        state["ybuf"][l][:, p] = y[b, state["win"][:, 3:]].float().cpu()
    return hook


for l in LAYERS:
    tm.layers[l].self_attn.o_proj.linear.register_forward_pre_hook(x_hook(l))
    tm.layers[l].post_attention_convolution.register_forward_hook(y_hook(l))

conv = {l: tm.layers[l].post_attention_convolution.conv1d.weight.detach().float().squeeze(1).cpu() for l in LAYERS}
torch.save(conv, f"{OUT}/conv.pt")

its = items()
NT = len(its)
X = {l: torch.zeros(NT, NPASS[l], WIN, 3072, dtype=torch.float16) for l in LAYERS}
YV = {l: None for l in LAYERS}
VALID = torch.zeros(NT, NPOS, dtype=torch.bool)
gens = [None] * NT
base = [prompt_ids(proc, t, e) + PREFIX for t, _, e in its]
batches = budget_batches([len(b) + NGEN for b in base], 96, 9000)
done = 0
with torch.no_grad():
    for idx in batches:
        seqs = [base[i] for i in idx]
        state["on"] = False
        ids, mask = left_pad(seqs, PAD)
        out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=NGEN,
                             pad_token_id=PAD, eos_token_id=END_MSG)
        full, tlen = [], []
        for r, i in enumerate(idx):
            th = []
            for t in out[r, ids.shape[1]:].tolist():
                if t in (END_MSG, PAD):
                    break
                th.append(t)
            gens[i] = th
            full.append(seqs[r] + th + [PAD] * (NGEN - len(th)))   # right-fill so the window always exists
            tlen.append(len(th))
        ids, mask = left_pad(full, PAD)
        for r in range(len(idx)):
            if tlen[r] < NGEN:
                mask[r, ids.shape[1] - (NGEN - tlen[r]):] = 0
        T = ids.shape[1]
        win = torch.zeros(len(idx), WIN, dtype=torch.long)
        for r, i in enumerate(idx):
            L = len(seqs[r])
            off = T - len(full[r])
            assert full[r][L - 3] == END_MSG and full[r][L - 1] == PREFIX[1]
            start = off + L - 4 - 3
            win[r] = torch.arange(start, start + WIN)
            VALID[i, :4] = True
            VALID[i, 4:4 + tlen[r]] = True
        state["win"] = win.cuda()
        state["xbuf"] = {l: torch.zeros(len(idx), NPASS[l], WIN, 3072, dtype=torch.float16) for l in LAYERS}
        state["ybuf"] = {l: torch.zeros(len(idx), NPASS[l], NPOS, 2304) for l in LAYERS}
        state["on"] = True
        model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
        state["on"] = False
        for l in LAYERS:
            X[l][idx] = state["xbuf"][l]
            sel = [r for r, i in enumerate(idx) if i < NVER]
            if sel:
                if YV[l] is None:
                    YV[l] = torch.zeros(NVER, NPASS[l], NPOS, 2304)
                YV[l][[idx[r] for r in sel]] = state["ybuf"][l][sel]
        done += len(idx)
        torch.cuda.empty_cache()
        log("cap", f"{done}/{NT}")

meta = {"sets": [s for _, s, _ in its], "efforts": [e for _, _, e in its], "gens": gens, "N": N, "NGEN": NGEN,
        "WIN": WIN, "layers": LAYERS}
json.dump(meta, open(f"{OUT}/meta.json", "w"))
torch.save(VALID, f"{OUT}/valid.pt")
torch.save(YV, f"{OUT}/yver.pt")
for l in LAYERS:
    torch.save(X[l], f"{OUT}/L{l}.pt")
log("CAPTURE_DONE", NT)
