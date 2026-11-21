"""Fidelity harness (HF only): KL(original || candidate) and top-1 agreement on harmless val prompts.

Pod layout: this file, cands.py, hf_name_map.json in /root/fid; hf_load.py in /root/parity;
data/val/harmless.jsonl at /root/fid/data/harmless.jsonl; big cache in /srv/fid.

  python kl_eval.py gen                      # 200 seeded val ids + orig greedy continuations (effort 0, <=256 tokens)
  python kl_eval.py cache                    # orig teacher-forced log-probs (fp16, full vocab) + fp32 argmax
  python kl_eval.py eval <cand_dir|orig> ... [--bs 8] [--order sorted|shuffle] [--tag x]

Definitions
- Continuation = the original's full greedy output (thinking included), truncated after eos, at most 256 tokens.
- For each continuation position t (predicting cont[t] from prompt + cont[:t]): KL_t = sum_v p_v (log p_v - log q_v),
  p = original, q = candidate, both from logits in fp32 (the cached original log-probs are stored in fp16 and
  renormalised in fp32 before use).
- KL = mean of KL_t over all positions of all 200 prompts (token-pooled). top1 = share of positions where the
  candidate's argmax equals the original's (fp32) argmax. 95% CI: bootstrap over prompts (10,000 resamples, seed 1234),
  pooled ratio within each resample.
- Engine: PR loader, experts dequantised to BF16, BF16 activations (W4A16). act_quant OFF (AQ=0 default).
- Candidates are always scored with the cache's batching (bs 8, sorted by length, right padding), so orig vs its
  own cache is exactly 0 and the KL measures only the edit.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/root/parity")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_load import encode, load  # noqa: E402

FID = os.environ.get("FID_DIR", "/root/fid")
BIG = os.environ.get("FID_BIG", "/srv/fid")
DATA = f"{FID}/data/harmless.jsonl"
IDS_PATH = f"{FID}/kl_val_ids.json"
CONT_PATH = f"{FID}/orig_cont_val.pt"
CTAG = os.environ.get("CACHE_TAG", "")  # alternate reference cache (e.g. "_aq0" for the no-act-quant diagnostic)
LP_PATH = f"{BIG}/orig_lp{CTAG}.f16"
META_PATH = f"{FID}/orig_cache_meta{CTAG}.json"
TOP1_PATH = f"{FID}/orig_top1{CTAG}.npy"
RESULTS = f"{FID}/kl_results.jsonl"
N_PROMPTS, MAX_NEW, SEED, EOS = 200, 256, 1234, 200006
AQ = os.environ.get("AQ", "0") == "1"  # OFF by decision 11:05Z (W4A16, PR loader); see fidelity/CONFIG.md


def pick_ids() -> list[dict]:
    rows = [json.loads(l) for l in open(DATA)]
    idx = random.Random(SEED).sample(range(len(rows)), N_PROMPTS)
    return [rows[i] for i in idx]


@torch.no_grad()
def cmd_gen(model, proc, bs: int = 100) -> None:
    rows = pick_ids()
    json.dump([r["id"] for r in rows], open(IDS_PATH, "w"), indent=0)
    tok = proc.tokenizer
    prompts = [encode(proc, r["text"], effort=0, device="cpu")[0].tolist() for r in rows]
    conts: list[list[int]] = [None] * len(rows)  # type: ignore[list-item]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))  # similar lengths per batch: less padding
    pad = tok.pad_token_id if tok.pad_token_id is not None else 199999
    t0 = time.time()
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        L = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):  # left padding
            ids[j, L - len(prompts[i]):] = torch.tensor(prompts[i])
            mask[j, L - len(prompts[i]):] = 1
        out = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), do_sample=False, temperature=None,
                             top_p=None, top_k=None, max_new_tokens=MAX_NEW, eos_token_id=EOS, pad_token_id=pad)
        for j, i in enumerate(idx):
            c = out[j, L:].tolist()
            if EOS in c:
                c = c[:c.index(EOS) + 1]
            conts[i] = c
        print(f"gen batch {b // bs}: {len(idx)} prompts, {time.time() - t0:.0f}s", flush=True)
    torch.save({"ids": [r["id"] for r in rows], "prompt_ids": prompts, "cont": conts, "effort": 0,
                "max_new_tokens": MAX_NEW, "act_quant": AQ, "batch": bs, "padding": "left"}, CONT_PATH)
    lens = [len(c) for c in conts]
    print(f"GEN_DONE {time.time() - t0:.0f}s; cont len mean {np.mean(lens):.1f}, eos-terminated "
          f"{sum(c[-1] == EOS for c in conts)}/{len(conts)}, total positions {sum(lens)}", flush=True)


def batches(n: int, bs: int, order: str, lens: list[int]) -> list[list[int]]:
    if order == "sorted":
        idx = sorted(range(n), key=lambda i: -lens[i])
    else:
        idx = list(range(n))
        random.Random(int(order.replace("shuffle", "") or 1)).shuffle(idx)
    return [idx[b:b + bs] for b in range(0, n, bs)]


@torch.no_grad()
def tf_logits(model, prompts: list[list[int]], conts: list[list[int]], idx: list[int]):
    """Right-padded teacher forcing. Yields (i, fp32 logits [L_i, V]) for the continuation positions."""
    seqs = [prompts[i] + conts[i] for i in idx]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(idx), T), 199999, dtype=torch.long)
    mask = torch.zeros((len(idx), T), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(s)
        mask[j, :len(s)] = 1
    out = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
    h = out.last_hidden_state
    for j, i in enumerate(idx):
        p, L = len(prompts[i]), len(conts[i])
        logits = model.logit_softcap(model.lm_head(h[j, p - 1:p - 1 + L]).float())
        yield i, logits


def load_cont():
    d = torch.load(CONT_PATH)
    return d["prompt_ids"], d["cont"]


@torch.no_grad()
def cmd_cache(model, bs: int = 8) -> None:
    prompts, conts = load_cont()
    lens = [len(c) for c in conts]
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    V = model.lm_head.out_features
    os.makedirs(BIG, exist_ok=True)
    lp = np.memmap(LP_PATH, dtype=np.float16, mode="w+", shape=(int(offs[-1]), V))
    top1 = np.zeros(int(offs[-1]), dtype=np.int64)
    self_agree = 0
    t0 = time.time()
    for idx in batches(len(prompts), bs, "sorted", lens):
        for i, logits in tf_logits(model, prompts, conts, idx):
            l = logits.log_softmax(-1)
            lp[offs[i]:offs[i + 1]] = l.half().cpu().numpy()
            a = logits.argmax(-1)
            top1[offs[i]:offs[i + 1]] = a.cpu().numpy()
            self_agree += int((a.cpu() == torch.tensor(conts[i])).sum())
    lp.flush()
    np.save(TOP1_PATH, top1)
    meta = {"n_prompts": len(prompts), "n_pos": int(offs[-1]), "V": V, "offs": offs.tolist(), "bs": bs,
            "order": "sorted", "act_quant": AQ, "dtype": "float16 log_softmax", "path": LP_PATH,
            "tf_argmax_eq_greedy_token": self_agree / int(offs[-1]), "sec": round(time.time() - t0, 1)}
    json.dump(meta, open(META_PATH, "w"))
    print(f"CACHE_DONE {meta['sec']}s, positions {meta['n_pos']}, TF argmax == greedy token "
          f"{meta['tf_argmax_eq_greedy_token']:.4f}", flush=True)


def boot_ci(num: np.ndarray, den: np.ndarray, n: int = 10000) -> list[float]:
    rng = np.random.default_rng(SEED)
    k = len(num)
    s = rng.integers(0, k, size=(n, k))
    vals = num[s].sum(1) / den[s].sum(1)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


@torch.no_grad()
def eval_one(model, prompts, conts, lp_all, top1_all, offs, bs: int, order: str) -> dict:
    n = len(prompts)
    kl_sum, agree, cont_agree, cnt = (np.zeros(n) for _ in range(4))
    pos_kl: list[np.ndarray] = []
    lens = [len(c) for c in conts]
    t0 = time.time()
    for idx in batches(n, bs, order, lens):
        for i, logits in tf_logits(model, prompts, conts, idx):
            a, b = int(offs[i]), int(offs[i + 1])
            logq = logits.log_softmax(-1)
            logp = torch.from_numpy(lp_all[a:b]).cuda().float()
            logp = logp - logp.logsumexp(-1, keepdim=True)
            kl = (logp.exp() * (logp - logq)).sum(-1)
            kl_sum[i] = float(kl.sum())
            pos_kl.append(kl.cpu().numpy())
            am = logits.argmax(-1).cpu()
            agree[i] = float((am == torch.from_numpy(top1_all[a:b])).sum())
            cont_agree[i] = float((am == torch.tensor(conts[i])).sum())
            cnt[i] = b - a
    sec = time.time() - t0
    pk = np.sort(np.concatenate(pos_kl))[::-1]
    top = max(1, len(pk) // 100)
    pos_stats = {"median": float(np.median(pk)), "p99": float(np.percentile(pk, 99)), "max": float(pk[0]),
                 "top1pct_share": float(pk[:top].sum() / max(pk.sum(), 1e-30)),
                 "top01pct_share": float(pk[:max(1, len(pk) // 1000)].sum() / max(pk.sum(), 1e-30))}
    return {"pos_stats": pos_stats, "kl": float(kl_sum.sum() / cnt.sum()), "kl_ci": boot_ci(kl_sum, cnt),
            "top1": float(agree.sum() / cnt.sum()), "top1_ci": boot_ci(agree, cnt),
            "top1_vs_cont": float(cont_agree.sum() / cnt.sum()), "n": n, "n_pos": int(cnt.sum()),
            "tf_sec": round(sec, 1), "per_prompt_kl": (kl_sum / cnt).round(6).tolist()}


def cmd_eval(model, targets: list[str], bs: int, order: str, tag: str) -> None:
    import cands
    prompts, conts = load_cont()
    meta = json.load(open(META_PATH))
    offs = np.array(meta["offs"])
    t0 = time.time()
    lp_all = np.fromfile(LP_PATH, dtype=np.float16).reshape(meta["n_pos"], meta["V"])
    top1_all = np.load(TOP1_PATH)
    print(f"cache loaded in {time.time() - t0:.0f}s", flush=True)
    for tgt in targets:
        t1 = time.time()
        if tgt == "orig":
            cands.restore(model)
            cid = "orig"
        else:
            info = cands.load_candidate(model, tgt)
            cid = info["cand_id"]
        r = eval_one(model, prompts, conts, lp_all, top1_all, offs, bs, order)
        cands.restore(model)
        r.update({"cand": cid, "tag": tag, "bs": bs, "order": order, "act_quant": AQ,
                  "total_sec": round(time.time() - t1, 1), "ref": {"bs": meta["bs"], "order": meta["order"]}})
        with open(RESULTS, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"RESULT {cid} [{tag}] KL {r['kl']:.5f} {r['kl_ci']} top1 {r['top1']:.4f} {r['top1_ci']} "
              f"top1_vs_cont {r['top1_vs_cont']:.4f} sec {r['total_sec']} pos {r['pos_stats']}", flush=True)


@torch.no_grad()
def cmd_routediag(model, n_batches: int = 5) -> None:
    """Router top-8 expert sets per (real token, visit): cache batching (bs 8, sorted) vs batch size 1,
    on the first n_batches*8 prompts of the sorted order. Reports how often the set changes."""
    prompts, conts = load_cont()
    lens = [len(c) for c in conts]
    sel = [i for b in batches(len(prompts), 8, "sorted", lens)[:n_batches] for i in b]
    store: list[torch.Tensor] = []
    hooks = [m.register_forward_hook(lambda _m, _i, out: store.append(out[2].sort(-1).values.cpu()))
             for name, m in model.named_modules() if type(m).__name__ == "Inkling2LoopTopKRouter"]

    def run(idx_groups):
        sets = {}
        for idx in idx_groups:
            store.clear()
            seqs = [prompts[i] + conts[i] for i in idx]
            T = max(len(s) for s in seqs)
            for _ in tf_logits(model, prompts, conts, idx):
                pass
            for v, ix in enumerate(store):  # visit order; ix [B*T, 8]
                ix = ix.view(len(idx), T, -1)
                for j, i in enumerate(idx):
                    sets[(i, v)] = ix[j, :len(seqs[j])]
        return sets

    ref = run([sel[k:k + 8] for k in range(0, len(sel), 8)])
    one = run([[i] for i in sel])
    for h in hooks:
        h.remove()
    n_visits = max(v for _, v in ref) + 1
    tv_flip = tv_tot = 0
    tok_any = tok_tot = 0
    first_visit = np.zeros(n_visits)
    for i in sel:
        flips = torch.stack([(ref[(i, v)] != one[(i, v)]).any(-1) for v in range(n_visits)])  # [V, T]
        tv_flip += int(flips.sum())
        tv_tot += flips.numel()
        anyf = flips.any(0)
        tok_any += int(anyf.sum())
        tok_tot += anyf.numel()
        for t in torch.nonzero(anyf).flatten().tolist():
            first_visit[int(torch.nonzero(flips[:, t])[0])] += 1
    out = {"prompts": len(sel), "router_calls_per_forward": n_visits, "token_visit_flip_rate": tv_flip / tv_tot,
           "token_any_flip_rate": tok_any / tok_tot, "first_flip_visit_hist": first_visit.astype(int).tolist()}
    json.dump(out, open(f"{FID}/routediag{CTAG}.json", "w"))
    print("ROUTEDIAG", json.dumps(out), flush=True)


def main() -> None:
    supp = "--supplementary" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--supplementary"]
    if AQ and not supp:
        raise SystemExit("act_quant is OFF for the harness (W4A16). AQ=1 needs --supplementary.")
    if supp:
        os.environ["FID_SUPPLEMENTARY"] = "1"
    cmd = args[0]
    opts = {"--bs": None, "--order": "sorted", "--tag": ""}
    rest = []
    i = 1
    while i < len(args):
        if args[i] in opts:
            opts[args[i]] = args[i + 1]
            i += 2
        else:
            rest.append(args[i])
            i += 1
    t0 = time.time()
    model, proc = load(act_quant=AQ)
    if AQ:
        from fast_aq import enable_fast_aq
        enable_fast_aq(model)  # bit-identical to hf_load's reference W4A4 path (scripts/fidelity/test_fast_aq.py)
    print(f"loaded {time.time() - t0:.0f}s act_quant={AQ}", flush=True)
    if cmd == "gen":
        cmd_gen(model, proc, int(opts["--bs"] or 100))
    elif cmd == "cache":
        cmd_cache(model, int(opts["--bs"] or 8))
    elif cmd == "eval":
        cmd_eval(model, rest, int(opts["--bs"] or 8), opts["--order"], opts["--tag"])
    elif cmd == "routediag":
        cmd_routediag(model)
    elif cmd == "all":  # gen, cache, noise floor, baselines in one process
        cmd_gen(model, proc, int(opts["--bs"] or 100))
        cmd_cache(model, 8)
        cmd_eval(model, ["orig"], 8, "sorted", "det_same_batching")
        cmd_eval(model, ["orig"], 1, "sorted", "noise_bs1")
        cmd_eval(model, ["orig"], 8, "shuffle1", "noise_shuffle_bs8")
        cmd_eval(model, rest, 8, "sorted", "baseline")
        cmd_routediag(model)
    print(f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
