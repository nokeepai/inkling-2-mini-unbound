"""Check fast_aq gives the same logits as hf_load's reference W4A4 path; time both."""
import json
import sys
import time
import types

import torch

sys.path.insert(0, "/root/parity")
sys.path.insert(0, "/root/fid")
from hf_load import _experts_forward_aq, encode, load  # noqa: E402
from fast_aq import enable_fast_aq  # noqa: E402

model, proc = load(act_quant=True)
rows = [json.loads(l) for l in open("/root/fid/data/harmless.jsonl")][:4]
seqs = [encode(proc, r["text"], 0, device="cpu")[0].tolist() for r in rows]
T = max(map(len, seqs))
ids = torch.full((4, T), 199999)
m = torch.zeros_like(ids)
for j, s in enumerate(seqs):
    ids[j, :len(s)] = torch.tensor(s)
    m[j, :len(s)] = 1
ids, m = ids.cuda(), m.cuda()


def run():
    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad():
        lg = model(input_ids=ids, attention_mask=m, use_cache=False).logits
    torch.cuda.synchronize()
    return lg, time.time() - t


def set_ref():
    for mod in model.modules():
        if type(mod).__name__ == "Inkling2LoopExperts":
            mod.forward = types.MethodType(_experts_forward_aq, mod)


def decode(n=24):
    g = encode(proc, rows[0]["text"], 0)
    t = time.time()
    with torch.no_grad():
        o = model.generate(input_ids=g, attention_mask=torch.ones_like(g), do_sample=False, max_new_tokens=n,
                           temperature=None, top_p=None, top_k=None)
    return o, (time.time() - t) / n


run()
ref, tr = run()
o_ref, sr = decode()
enable_fast_aq(model)
run()
fast, tf = run()
o_fast, sf = decode()
d = (ref - fast).abs()
print(f"prefill 4 seqs: ref {tr:.2f}s fast {tf:.2f}s; max|dlogit| {d.max().item():.3g}; exact {torch.equal(ref, fast)}")
print(f"decode bs1: ref {sr:.3f}s/step fast {sf:.3f}s/step; same tokens {torch.equal(o_ref, o_fast)}")
print("TEST_DONE")
