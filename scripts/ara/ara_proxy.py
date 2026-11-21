"""ARA fast proxy (agent ara, ink-research-1). TRAIN split, prompts held out from the capture set. Diagnostic only:
never used for selection in place of fleet/kl_worker rows.

Loads the HF model once, then for every /root/ara/jobs/<cid>/build/tensors.safetensors without proxy.json:
  swaps the wo_ud tensors in, measures, restores. Measures:
  - kl_proxy: KL(orig || cand) over 48 held-out harmless prompts (effort 0), teacher-forced on prompt + forced prefix +
    the original's 48 greedy tokens, mean over those 49 positions.
  - ref_score[v]: harmful held-out prompts (32 at e0, 32 at e0.9): projection of the decoder output at visit v (41 = pass-3
    L17, 47 = L23) at <|content_thinking|> onto the train refusal DoM there, rescaled so orig = 1 and harmless mean = 0.
    It includes what the MoE writes downstream of the edit.
"""
import glob
import json
import os
import random
import sys
import time

import torch
from safetensors.torch import load_file

sys.path.insert(0, "/root/diag")
from diaglib import END_MSG, PREFIX, left_pad, load, log, prompt_ids, read_jsonl, text_model  # noqa: E402

model, proc = load()
tok = proc.tokenizer
PAD = tok.pad_token_id if tok.pad_token_id is not None else 200006
tm = text_model(model)
meta = json.load(open("/root/ara/cap/meta.json"))
harmful = read_jsonl("/root/ara/data/harmful.jsonl")
harmless = read_jsonl("/root/ara/data/harmless.jsonl")
# held out: rows not drawn by the capture (capture used the first 256 of seeded shuffles; take from the tail)
rng = random.Random(777)
hf_rows = harmful[:]
rng.shuffle(hf_rows)
hl_rows = harmless[:]
rng.shuffle(hl_rows)
cap_texts = set()
for e, seed in ((0.0, 1234), (0.9, 4321)):
    r2 = random.Random(seed)
    hl = list(harmless)
    r2.shuffle(hl)
    cap_texts |= {r["text"] for r in hl[:256]}
hl_hold = [r["text"] for r in hl_rows if r["text"] not in cap_texts][:48]
hf_hold = [r["text"] for r in hf_rows][-64:]   # harmful capture was stratified; overlap possible, reported as caveat
VIS = {41: (17, 2), 47: (23, 0)}
D = torch.load("/root/diag/out/directions.pt", weights_only=False)
dirs = {}
DM = D["means"][0.0] if 0.0 in D["means"] else D["means"][[k for k in D["means"] if float(k) == 0.0][0]]
for v in VIS:
    mh = DM["out"]["harmful"][v, 3].float()
    ml = DM["out"]["harmless"][v, 3].float()
    d = (mh - ml) / (mh - ml).norm()
    dirs[v] = d.cuda()
calls = {}
grab = {}


def reset(*_a, **_k):
    calls.clear()


tm.register_forward_pre_hook(reset)


def mk(layer):
    def hook(_m, _i, out):
        c = calls.get(layer, 0)
        calls[layer] = c + 1
        for v, (l, p) in VIS.items():
            if l == layer and p == c:
                t = out[0] if isinstance(out, tuple) else out
                grab[v] = t[:, -1].float()
    return hook


for l in {l for l, _ in VIS.values()}:
    tm.layers[l].register_forward_hook(mk(l))

with torch.no_grad():
    hl_ids = [prompt_ids(proc, t, 0.0) + PREFIX for t in hl_hold]
    ids, mask = left_pad(hl_ids, PAD)
    out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=48, pad_token_id=PAD,
                         eos_token_id=END_MSG)
    kl_ids, kl_mask = out, (out != PAD).long()
    kl_mask[:, :ids.shape[1]] = mask
    NK = out.shape[1] - ids.shape[1] + 1
    hf_seqs = [prompt_ids(proc, t, e) + PREFIX for t in hf_hold[:32] for e in (0.0,)] + \
              [prompt_ids(proc, t, 0.9) + PREFIX for t in hf_hold[32:]]
    hf_ids, hf_mask = left_pad(hf_seqs, PAD)
    hl2 = [prompt_ids(proc, t, 0.0) + PREFIX for t in hl_hold[:32]]
    hl2_ids, hl2_mask = left_pad(hl2, PAD)


@torch.no_grad()
def measure():
    lg = model(input_ids=kl_ids, attention_mask=kl_mask, use_cache=False, logits_to_keep=NK + 1).logits[:, :-1].float()
    lp = torch.log_softmax(lg, -1)
    model(input_ids=hf_ids, attention_mask=hf_mask, use_cache=False, logits_to_keep=1)
    proj_hf = {v: (grab[v] @ dirs[v]) for v in VIS}
    model(input_ids=hl2_ids, attention_mask=hl2_mask, use_cache=False, logits_to_keep=1)
    proj_hl = {v: (grab[v] @ dirs[v]).mean() for v in VIS}
    return lp, proj_hf, proj_hl


lp0, ph0, pl0 = measure()
m0 = (kl_mask[:, -NK:] > 0).float()
log("baseline ready", {v: (float(ph0[v].mean()), float(pl0[v])) for v in VIS})
done = set()
while True:
    for f in sorted(glob.glob("/root/ara/jobs/*/build/tensors.safetensors")):
        d = f.split("/build/")[0]
        if d in done or os.path.exists(f"{d}/proxy.json"):
            continue
        time.sleep(2)
        T = load_file(f)
        saved = {}
        for n, t in T.items():
            l = int(n.split(".")[3])
            w = tm.layers[l].self_attn.o_proj.linear.weight
            saved[l] = w.detach().clone()
            w.data.copy_(t.to(w.device, w.dtype))
        lp, ph, pl = measure()
        for l, w0 in saved.items():
            tm.layers[l].self_attn.o_proj.linear.weight.data.copy_(w0)
        kl = ((lp0.exp() * (lp0 - lp)).sum(-1) * m0).sum() / m0.sum()
        res = {"kl_proxy": round(float(kl), 5)}
        for v in VIS:
            res[f"ref_score_v{v}"] = round(float((ph[v].mean() - pl0[v]) / (ph0[v].mean() - pl0[v])), 4)
            res[f"ref_score_v{v}_e0"] = round(float((ph[v][:32].mean() - pl0[v]) / (ph0[v][:32].mean() - pl0[v])), 4)
            res[f"harmless_shift_v{v}"] = round(float((pl[v] - pl0[v]) / (ph0[v].mean() - pl0[v])), 4)
        json.dump(res, open(f"{d}/proxy.json", "w"))
        done.add(d)
        log(os.path.basename(d), res)
    time.sleep(5)
