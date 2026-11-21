"""HF side of the parity check (run on ink-research-1, SGLang stopped).

  python hf_run.py greedy            -> /root/parity/hf_greedy.json, /root/parity/hf_visits.pt
  python hf_run.py tf <sglang.json>  -> /root/parity/hf_tf_<name>.json (teacher-forced top-1 on SGLang tokens)
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/root/parity")
from hf_load import greedy, load, teacher_forced_top1, text_model  # noqa: E402

AQ = os.environ.get("AQ", "0") == "1"  # emulate W4A4 expert activation quant (see hf_load)
SUF = "_aq" if AQ else ""
N_HIDDEN = 5  # prompts whose per-visit hidden states are dumped
P = json.load(open("/root/parity/prompts.json"))


def visit_hooks(model, store: list):
    """Record every decoder-layer call (48 visits) in call order, prefill only."""
    handles = []
    for idx, layer in enumerate(text_model(model).layers):
        def hook(_m, _i, out, idx=idx):
            t = out[0] if isinstance(out, tuple) else out
            store.append((idx, t.detach()[0].float().cpu()))
        handles.append(layer.register_forward_hook(hook))
    return handles


def run_greedy(model):
    out, visits = [], {}
    for r in P:
        ids = torch.tensor([r["input_ids"]], device="cuda:0")
        t = time.time()
        toks = greedy(model, ids, 64)
        out.append({"i": r["i"], "tokens": toks, "sec": round(time.time() - t, 2)})
        print(r["i"], len(toks), out[-1]["sec"], flush=True)
    json.dump(out, open(f"/root/parity/hf_greedy{SUF}.json", "w"))
    for r in P[:N_HIDDEN]:
        store: list = []
        hs = visit_hooks(model, store)
        with torch.no_grad():
            ids = torch.tensor([r["input_ids"]], device="cuda:0")
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        for h in hs:
            h.remove()
        visits[r["i"]] = store  # list of (physical_layer, [T, H]) in call order
    torch.save(visits, f"/root/parity/hf_visits{SUF}.pt")


def run_tf(model, path):
    sg = json.load(open(path))
    res = []
    for s in sg:
        r = P[s["i"]]
        ids = torch.tensor([r["input_ids"]], device="cuda:0")
        full = torch.cat([ids, torch.tensor([s["tokens"]], device=ids.device)], dim=1)
        with torch.no_grad():
            lg = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False).logits
        p = ids.shape[1]
        lp = lg[0, p - 1: p - 1 + len(s["tokens"])].float().log_softmax(-1)
        top2 = lp.topk(2, dim=-1)
        top1 = top2.indices[:, 0].tolist()
        margin = (top2.values[:, 0] - top2.values[:, 1]).tolist()
        # HF logprob gap between its own top-1 and the token SGLang chose, at positions where they differ
        chosen = lp.gather(1, torch.tensor(s["tokens"], device=lp.device)[:, None])[:, 0]
        gap = (top2.values[:, 0] - chosen).tolist()
        agree = sum(int(a == b) for a, b in zip(top1, s["tokens"]))
        res.append({"i": s["i"], "hf_top1": top1, "agree": agree, "n": len(top1),
                    "hf_margin": [round(x, 4) for x in margin],
                    "miss_gaps": [round(g, 4) for g, a, b in zip(gap, top1, s["tokens"]) if a != b]})
    name = os.path.basename(path).replace(".json", "")
    json.dump(res, open(f"/root/parity/hf_tf{SUF}_{name}.json", "w"))
    a, n = sum(x["agree"] for x in res), sum(x["n"] for x in res)
    gaps = sorted(g for x in res for g in x["miss_gaps"])
    print(f"TF top1 agreement {a}/{n} = {a / max(n, 1):.4f}; HF logprob gap at misses: {gaps}")


if __name__ == "__main__":
    t0 = time.time()
    model, _ = load(act_quant=AQ)
    print(f"loaded in {time.time() - t0:.0f}s; mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    if sys.argv[1] == "greedy":
        run_greedy(model)
    else:
        run_tf(model, sys.argv[2])
    print(f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; total {time.time() - t0:.0f}s")
