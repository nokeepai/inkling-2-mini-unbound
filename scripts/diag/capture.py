"""Task 1: capture per-visit states at key positions. Usage: python capture.py <effort> <set> [<set> ...]
Sets: harmful, harmless (seeded sample, same size as harmful), math, code, honest, crisis.
Positions (dim 2): 0 last instruction token, 1 <|end_message|>, 2 <|message_model|>, 3 <|content_thinking|>,
4 mean over the first <=16 greedy thinking tokens (teacher-forced from the model's own greedy start).
Kinds: out (decoder-layer output = partial block sum), attn (post_attention_convolution output),
mlp (post_mlp_convolution output), mlpin (mlp_input output, what the MoE reads before the message gate).
Saves fp16 memmaps [N, 48, 5, 2304] to /root/diag/cap/<set>_e<effort>_<kind>.npy plus meta json."""
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/root/diag")
from diaglib import budget_batches, END_MSG, PREFIX, VisitCounter, left_pad, load, log, prompt_ids, read_jsonl, text_model

EFFORT = float(sys.argv[1])
SETS = sys.argv[2:]
NGEN = 16
BS = int(os.environ.get("BS", "96"))
KINDS = ("out", "attn", "mlp", "mlpin")
CAP = "/root/diag/cap"
os.makedirs(CAP, exist_ok=True)


LIMIT = int(os.environ.get("LIMIT", "100000"))


def strat(rows, n, seed=1234):
    """Seeded sample of n rows, stratified by category (proportional, at least 1 per category)."""
    if n >= len(rows):
        return rows
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r)
    out = []
    for c, rs in sorted(by.items()):
        rng.shuffle(rs)
        out += rs[:max(1, round(n * len(rs) / len(rows)))]
    rng.shuffle(out)
    return out[:n]


def load_set(name):
    if name == "harmful":
        return [(r["text"], r["category"], r["id"]) for r in strat(read_jsonl("/root/diag/data/harmful.jsonl"), LIMIT)]
    if name == "harmless":
        rows = read_jsonl("/root/diag/data/harmless.jsonl")
        n = min(LIMIT, len(read_jsonl("/root/diag/data/harmful.jsonl")))
        random.Random(1234).shuffle(rows)
        return [(r["text"], "harmless", r["id"]) for r in rows[:n]]
    if name == "math":
        return [(q, "math", f"gsm{i}") for i, q in enumerate(json.load(open("/root/diag/data/gsm_q.json")))][:50]
    atoms = json.load(open("/root/diag/atoms_prompts.json"))
    return [(t, name, f"{name}{i}") for i, t in enumerate(atoms[name])]


model, proc = load()
tok = proc.tokenizer
PAD = tok.pad_token_id if tok.pad_token_id is not None else 200006
tm = text_model(model)
vc = VisitCounter(tm)
cur = {"pos": None, "tmask": None, "buf": None}


def gather(t, kind, v):
    t = t[0] if isinstance(t, tuple) else t
    t = t.float()
    b = torch.arange(t.shape[0], device=t.device)[:, None]
    pts = t[b, cur["pos"]]  # [B,4,D]
    tm_ = cur["tmask"]  # [B,T]
    th = (t * tm_[..., None]).sum(1) / tm_.sum(1, keepdim=True).clamp(min=1)
    cur["buf"][kind][:, v, :4] = pts.half().cpu()
    cur["buf"][kind][:, v, 4] = th.half().cpu()


def mk(layer, sub, kind):
    def hook(_m, _i, out):
        gather(out, kind, vc.visit(layer, kind))
    mod = tm.layers[layer] if sub is None else getattr(tm.layers[layer], sub)
    return mod.register_forward_hook(hook)


hooks = []
for i in range(24):
    hooks.append(mk(i, None, "out"))
    hooks.append(mk(i, "post_attention_convolution", "attn"))
    hooks.append(mk(i, "post_mlp_convolution", "mlp"))
    hooks.append(mk(i, "mlp_input", "mlpin"))
enabled = {"on": False}
_orig_hooks = list(hooks)


def set_hooks(on):
    # cheap toggle: hooks check a flag via cur["buf"] being None
    enabled["on"] = on


# wrap gather to no-op when disabled
_g = gather


def gather(t, kind, v):  # noqa: F811
    if enabled["on"]:
        _g(t, kind, v)


@torch.no_grad()
def run_set(name):
    items = load_set(name)
    base = [prompt_ids(proc, t, EFFORT) + PREFIX for t, _, _ in items]
    N = len(items)
    batches = budget_batches([len(b) + NGEN for b in base], BS, int(os.environ.get("TOK", "9000")))
    mm = {k: np.lib.format.open_memmap(f"{CAP}/{name}_e{EFFORT}_{k}.npy", mode="w+", dtype=np.float16,
                                       shape=(N, 48, 5, 2304)) for k in KINDS}
    gens = [None] * N
    done = 0
    for idx in batches:
        seqs = [base[i] for i in idx]
        enabled["on"] = False
        ids, mask = left_pad(seqs, PAD)
        t0 = time.time()
        ngen = NGEN if name in ("harmful", "harmless") else 0
        if ngen:
            out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=ngen,
                                 pad_token_id=PAD, eos_token_id=END_MSG)
        else:
            out = ids
        t1 = time.time()
        full, pos, tlen = [], [], []
        for r, i in enumerate(idx):
            g = out[r, ids.shape[1]:].tolist()
            th = []
            for x in g:
                if x in (END_MSG, PAD):
                    break
                th.append(x)
            gens[i] = tok.decode(th)
            full.append(seqs[r] + th)
            tlen.append(len(th))
        ids, mask = left_pad(full, PAD)
        T = ids.shape[1]
        P = torch.zeros(len(idx), 4, dtype=torch.long)
        tmask = torch.zeros(len(idx), T)
        for r in range(len(idx)):
            L = len(seqs[r])  # prompt + prefix
            off = T - len(full[r])
            # ... last_instr, END_MSG, MSG_MODEL, THINK
            P[r] = torch.tensor([off + L - 4, off + L - 3, off + L - 2, off + L - 1])
            assert full[r][L - 3] == END_MSG and full[r][L - 1] == PREFIX[1]
            tmask[r, off + L: off + L + tlen[r]] = 1
        cur["pos"] = P.cuda()
        cur["tmask"] = tmask.cuda()
        cur["buf"] = {k: torch.zeros(len(idx), 48, 5, 2304, dtype=torch.float16) for k in KINDS}
        enabled["on"] = True
        model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
        enabled["on"] = False
        log("t_gen", round(t1 - t0, 1), "t_fwd", round(time.time() - t1, 1), "T", T)
        for k in KINDS:
            mm[k][idx] = cur["buf"][k].numpy()
        done += len(idx)
        torch.cuda.empty_cache()
        log(name, EFFORT, f"{done}/{N}")
    for k in KINDS:
        mm[k].flush()
    meta = {"effort": EFFORT, "ids": [x[2] for x in items], "cats": [x[1] for x in items], "gens": gens}
    json.dump(meta, open(f"{CAP}/{name}_e{EFFORT}_meta.json", "w"))
    log("done", name, EFFORT)


for s in SETS:
    run_set(s)
log("ALL_DONE", EFFORT, SETS)
