"""ARA preserve-set capture (agent ara, ink-research-1). Train split only.
Harmless prompts (a fresh seeded draw of 192 per effort 0 / 0.9, disjoint from the main capture's harmless draw), the model's
own 40 greedy thinking tokens; records the wo_ud input on a window of 3 conv-context + 32 targets = thinking tokens 8..39
for L4-23, every pass. Saved /root/ara/cap/P{l}.pt fp16 [N, Pp, 35, 3072] and pvalid.pt [N, 32].
These broaden the preserve term beyond the 12 refusal-deciding positions (KL is measured deep into the reply)."""
import json
import random
import sys

import torch

sys.path.insert(0, "/root/diag")
from diaglib import END_MSG, PREFIX, budget_batches, left_pad, load, log, prompt_ids, read_jsonl, text_model  # noqa: E402

N = 192
NGEN, T0, NT = 40, 8, 32
WIN = 3 + NT
LAYERS = list(range(4, 24))
OUT = "/root/ara/cap"
harmless = read_jsonl("/root/ara/data/harmless.jsonl")
used = set()
for seed in (1234, 4321):
    r = random.Random(seed)
    hl = list(harmless)
    r.shuffle(hl)
    used |= {x["text"] for x in hl[:256]}
pool = [x["text"] for x in harmless if x["text"] not in used]
random.Random(99).shuffle(pool)
its = [(t, 0.0) for t in pool[:N]] + [(t, 0.9) for t in pool[N:2 * N]]

model, proc = load()
tok = proc.tokenizer
PAD = tok.pad_token_id if tok.pad_token_id is not None else 200006
tm = text_model(model)
NPASS = {l: (3 if 6 <= l <= 17 else 1) for l in LAYERS}
state = {"on": False, "calls": {}}
tm.register_forward_pre_hook(lambda *a, **k: state["calls"].clear())


def x_hook(l):
    def hook(_m, inp):
        if not state["on"]:
            return
        c = state["calls"].get(l, 0)
        state["calls"][l] = c + 1
        x = inp[0]
        b = torch.arange(x.shape[0], device=x.device)[:, None]
        state["xbuf"][l][:, c] = x[b, state["win"]].half().cpu()
    return hook


for l in LAYERS:
    tm.layers[l].self_attn.o_proj.linear.register_forward_pre_hook(x_hook(l))
X = {l: torch.zeros(len(its), NPASS[l], WIN, 3072, dtype=torch.float16) for l in LAYERS}
V = torch.zeros(len(its), NT, dtype=torch.bool)
base = [prompt_ids(proc, t, e) + PREFIX for t, e in its]
done = 0
with torch.no_grad():
    for idx in budget_batches([len(b) + NGEN for b in base], 64, 12000):
        seqs = [base[i] for i in idx]
        ids, mask = left_pad(seqs, PAD)
        out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=NGEN,
                             pad_token_id=PAD, eos_token_id=200006)
        full, tl = [], []
        for r in range(len(idx)):
            g = [t for t in out[r, ids.shape[1]:].tolist()]
            k = len(g)
            for j, t in enumerate(g):
                if t == PAD:
                    k = j
                    break
            g = g[:k]
            tl.append(k)
            full.append(seqs[r] + g + [PAD] * (NGEN - k))
        ids, mask = left_pad(full, PAD)
        Tn = ids.shape[1]
        win = torch.zeros(len(idx), WIN, dtype=torch.long)
        for r, i in enumerate(idx):
            if tl[r] < NGEN:
                mask[r, Tn - (NGEN - tl[r]):] = 0
            L = len(seqs[r])
            off = Tn - len(full[r])
            s = off + L + T0 - 3
            win[r] = torch.arange(s, s + WIN)
            V[i, :max(0, min(NT, tl[r] - T0))] = True
        state["win"] = win.cuda()
        state["xbuf"] = {l: torch.zeros(len(idx), NPASS[l], WIN, 3072, dtype=torch.float16) for l in LAYERS}
        state["on"] = True
        model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
        state["on"] = False
        for l in LAYERS:
            X[l][idx] = state["xbuf"][l]
        done += len(idx)
        log("pcap", f"{done}/{len(its)}")
torch.save(V, f"{OUT}/pvalid.pt")
for l in LAYERS:
    torch.save(X[l], f"{OUT}/P{l}.pt")
log("PCAP_DONE", float(V.float().mean()))
