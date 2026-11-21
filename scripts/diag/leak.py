"""Task 5: how much of OU's Q survives post_attention_convolution after the wo_ud weight edit.
Also a padding sanity test (batched left-padded generate vs single) and speed timings."""
import json
import math
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/diag")
from diaglib import (PREFIX, VisitCounter, left_pad, load, log, project_out_hooks, prompt_ids, read_jsonl,
                     text_model)

OUT = "/root/diag/leak.json"
model, proc = load()
tok = proc.tokenizer
PAD = tok.pad_token_id if tok.pad_token_id is not None else 200006
tm = text_model(model)
edit = torch.load("/root/diag/edit.pt", weights_only=True)
Q = edit["Q"].float()
s_attn = edit["s_attn"].float()
Qd = Q.cuda()

harm = read_jsonl("/root/diag/data/harmful.jsonl")
random.Random(0).shuffle(harm)
seqs = [prompt_ids(proc, h["text"], 0) + PREFIX for h in harm[:50]]
vc = VisitCounter(tm)
state = {"mask": None, "mode": None}
acc = {}


def add(key, v, val):
    acc.setdefault(key, [0.0] * 48)
    acc[key][v] += float(val)


def conv_cf(mod, h):
    """Re-apply the depthwise causal conv to h [B,T,D] (fp32): (conv(h)+h)/sqrt(5)."""
    w = mod.conv1d.weight.float()  # [D,1,4]
    x = h.float() * state["mask"][..., None]
    y = F.conv1d(x.transpose(1, 2), w, padding=3, groups=x.shape[-1])[..., : x.shape[1]].transpose(1, 2)
    return (y + x) / math.sqrt(5)


def make_hook(layer):
    mod = tm.layers[layer].post_attention_convolution

    def hook(_m, inp, out):
        v = vc.visit(layer, "pac")
        m = state["mask"][..., None].float()
        h = inp[0].float() * m
        y = out.float() * m
        mode = state["mode"]
        add(f"{mode}/Qy2", v, ((y @ Qd) ** 2).sum())
        add(f"{mode}/Qh2", v, ((h @ Qd) ** 2).sum())
        add(f"{mode}/y2", v, (y ** 2).sum())
        if mode == "orig":
            # local counterfactual: same input, Q removed (s=1) or reflected (OU s) before the conv
            ycheck = conv_cf(mod, h)
            add("orig/cf_check_err2", v, ((ycheck - y) ** 2).sum())
            for s_name, s in (("s1", 1.0), ("ou", float(s_attn[layer]))):
                hp = h - s * (h @ Qd) @ Qd.T
                yc = conv_cf(mod, hp)
                add(f"cf_{s_name}/Qy2", v, ((yc @ Qd) ** 2).sum())
                # deviation from the ideal (1-s) scaling of the Q component
                add(f"cf_{s_name}/dev2", v, (((yc - (1 - s) * y) @ Qd) ** 2).sum())
    return mod.register_forward_hook(hook)


handles = [make_hook(i) for i in range(24)]


@torch.no_grad()
def run(mode):
    state["mode"] = mode
    for b in range(0, 50, 25):
        ids, mask = left_pad(seqs[b:b + 25], PAD)
        state["mask"] = mask.float()
        model(input_ids=ids, attention_mask=mask, use_cache=False)
    log("ran", mode)


orig_w = {i: tm.layers[i].self_attn.o_proj.linear.weight.detach().clone() for i in range(24)}


def set_edit(s_vec):
    for i in range(24):
        W = orig_w[i]
        s = float(s_vec[i])
        if s == 0:
            tm.layers[i].self_attn.o_proj.linear.weight.data.copy_(W)
            continue
        W32 = W.float()
        Wn = (W32 - s * (Qd @ (Qd.T @ W32))).to(torch.bfloat16)
        tm.layers[i].self_attn.o_proj.linear.weight.data.copy_(Wn)


run("orig")
set_edit(s_attn)
run("ou_edit")
set_edit(torch.where(s_attn > 0, torch.ones_like(s_attn), s_attn))
run("s1_edit")
set_edit(torch.zeros(24))
for h in handles:
    h.remove()

# measuring hook order: re-run conv_hook with the projection registered BEFORE the measuring hook
handles = []
rm = project_out_hooks(model, Q, [i for i in range(24) if s_attn[i] > 0], ["post_attention_convolution"])
handles = [make_hook(i) for i in range(24)]
run("conv_hook_post")
rm()
for h in handles:
    h.remove()


def ratio(a, b):
    return [math.sqrt(x / y) if y > 0 else None for x, y in zip(acc[a], acc[b])]


res = {
    "n_prompts": 50, "effort": 0, "positions": "all prompt tokens + forced prefix",
    "s_attn": s_attn.tolist(),
    "local_leak_s1": ratio("cf_s1/Qy2", "orig/Qy2"),
    "local_ou_Qy_ratio": ratio("cf_ou/Qy2", "orig/Qy2"),
    "local_ou_dev_from_ideal": ratio("cf_ou/dev2", "orig/Qy2"),
    "e2e_ou_Qy_ratio": ratio("ou_edit/Qy2", "orig/Qy2"),
    "e2e_ou_Qh_ratio": ratio("ou_edit/Qh2", "orig/Qh2"),
    "e2e_s1_Qy_ratio": ratio("s1_edit/Qy2", "orig/Qy2"),
    "e2e_s1_Qh_ratio": ratio("s1_edit/Qh2", "orig/Qh2"),
    "conv_hook_Qy_ratio": ratio("conv_hook_post/Qy2", "orig/Qy2"),
    "orig_Qy_share": ratio("orig/Qy2", "orig/y2"),
    "cf_check_rel_err": ratio("orig/cf_check_err2", "orig/y2"),
}
json.dump(res, open(OUT, "w"), indent=1)
log("leak saved")

# ---- padding sanity + speed ----
test = [prompt_ids(proc, h["text"], 0) + PREFIX for h in harm[50:54]]
t0 = __import__("time").time()
with torch.no_grad():
    ids, mask = left_pad(test, PAD)
    out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=16, pad_token_id=PAD)
t_batch = __import__("time").time() - t0
batched = [out[i, ids.shape[1]:].tolist() for i in range(4)]
single = []
for s in test:
    x = torch.tensor([s], device="cuda:0")
    o = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=16,
                       pad_token_id=PAD)
    single.append(o[0, x.shape[1]:].tolist())
# big-batch timing
big = [prompt_ids(proc, h["text"], 0) + PREFIX for h in harm[:100]]
t0 = __import__("time").time()
with torch.no_grad():
    ids, mask = left_pad(big, PAD)
    model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=16, pad_token_id=PAD)
t_big = __import__("time").time() - t0
t0 = __import__("time").time()
with torch.no_grad():
    model(input_ids=ids, attention_mask=mask, use_cache=False)
t_fwd = __import__("time").time() - t0
pad_res = {"lens": [len(s) for s in test], "batched": batched, "single": single,
           "match": [a == b for a, b in zip(batched, single)], "t_batch4_16tok": t_batch,
           "t_batch100_16tok": t_big, "t_fwd100": t_fwd, "pad_id": PAD,
           "decoded": [tok.decode(b) for b in batched]}
json.dump(pad_res, open("/root/diag/padtest.json", "w"), indent=1)
log("padtest", pad_res["match"], t_batch, t_big, t_fwd)
