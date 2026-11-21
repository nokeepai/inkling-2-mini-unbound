"""HF capability inner loop: MMLU-300 by letter logits (effort 0) and GSM8K-100 (effort 0.5, thinking cap 1,536).
Prompt format and scoring rules: harness/cap/FORMAT.md (shared with the SGLang scorer).

  python hf_cap_eval.py mmlu  [cand_dir|orig]
  python hf_cap_eval.py gsm8k [cand_dir|orig] [--bs 100]
  python hf_cap_eval.py both  [cand_dir|orig]
Outputs: /root/fid/cap_results.jsonl (one row per run), /root/fid/cap_<bench>_<cand>.json (per-item).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/root/parity")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_load import load, render  # noqa: E402

FID = os.environ.get("FID_DIR", "/root/fid")
AQ = os.environ.get("AQ", "0") == "1"  # OFF by decision 11:05Z (W4A16)
MSG_MODEL, TEXT, THINK, END, EOS, PAD = 200001, 200004, 200008, 200010, 200006, 199999
THINK_CAP, ANSWER_CAP, SEED = 1536, 512, 1234
MMLU_SUFFIX = "Answer with the letter of the correct option only."
GSM_SUFFIX = 'Solve the problem. End your reply with a line of the form "Answer: <number>".'


def ids_for(proc, user: str, effort: float) -> list[int]:
    tok = proc.tokenizer
    return tok(render(proc, user, effort), add_special_tokens=False).input_ids


def boot(correct: np.ndarray, n: int = 10000) -> list[float]:
    rng = np.random.default_rng(SEED)
    s = rng.integers(0, len(correct), size=(n, len(correct)))
    v = correct[s].mean(1) * 100
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


@torch.no_grad()
def mmlu(model, proc, bs: int = 16) -> dict:
    rows = [json.loads(l) for l in open(f"{FID}/cap/mmlu300.jsonl")]
    tok = proc.tokenizer
    letters = []
    for L in "ABCD":
        t = tok.encode(L, add_special_tokens=False)
        assert len(t) == 1, (L, t)
        letters.append(t[0])
    seqs = []
    for r in rows:
        opts = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", r["choices"]))
        user = f"{r['question'].strip()}\n\n{opts}\n\n{MMLU_SUFFIX}"
        seqs.append(ids_for(proc, user, 0) + [MSG_MODEL, TEXT])
    preds = []
    t0 = time.time()
    for b in range(0, len(seqs), bs):
        chunk = seqs[b:b + bs]
        T = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), T), PAD, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, s in enumerate(chunk):
            ids[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
        h = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False).last_hidden_state
        last = torch.stack([h[j, len(s) - 1] for j, s in enumerate(chunk)])
        logits = model.logit_softcap(model.lm_head(last).float())
        lp = logits.log_softmax(-1)[:, letters]
        preds += lp.argmax(-1).tolist()
    correct = np.array([int(p == r["answer"]) for p, r in zip(preds, rows)], dtype=float)
    per = [{"id": r["id"], "pred": p, "answer": r["answer"]} for p, r in zip(preds, rows)]
    return {"metric": "mmlu300_acc", "value": float(correct.mean() * 100), "CI": boot(correct), "n": len(rows),
            "sec": round(time.time() - t0, 1), "letter_ids": letters, "_per": per}


def extract(answer_text: str) -> float | None:
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", answer_text)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", answer_text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def _left_pad(seqs: list[list[int]]):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for j, s in enumerate(seqs):
        ids[j, T - len(s):] = torch.tensor(s)
        mask[j, T - len(s):] = 1
    return ids.cuda(), mask.cuda(), T


@torch.no_grad()
def _gen(model, seqs: list[list[int]], max_new: int, bs: int) -> list[list[int]]:
    outs: list[list[int]] = []
    for b in range(0, len(seqs), bs):
        ids, mask, T = _left_pad(seqs[b:b + bs])
        out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, temperature=None, top_p=None,
                             top_k=None, max_new_tokens=max_new, eos_token_id=EOS, pad_token_id=PAD)
        for j in range(out.shape[0]):
            c = out[j, T:].tolist()
            if EOS in c:
                c = c[:c.index(EOS) + 1]
            outs.append(c)
    return outs


def _parse(gen: list[int]) -> dict:
    """Split a generation into thinking and answer parts (token lists) and flags."""
    think, ans, state = [], [], "pre"
    closed_think = ans_open = ans_closed = False
    for i, t in enumerate(gen):
        if state == "pre":
            if t == THINK:
                state = "think"
            elif t == TEXT:
                state = "answer"
                ans_open = True
        elif state == "think":
            if t == END:
                closed_think = True
                state = "between"
            else:
                think.append(t)
        elif state == "between":
            if t == TEXT:
                state = "answer"
                ans_open = True
        elif state == "answer":
            if t in (END, EOS):
                ans_closed = True
                break
            ans.append(t)
    return {"think": think, "answer": ans, "closed_think": closed_think, "ans_open": ans_open,
            "ans_closed": ans_closed or (gen and gen[-1] == EOS and ans_open)}


@torch.no_grad()
def gsm8k(model, proc, bs: int = 100) -> dict:
    rows = [json.loads(l) for l in open(f"{FID}/cap/gsm8k100.jsonl")]
    tok = proc.tokenizer
    prompts = [ids_for(proc, f"{r['question'].strip()}\n\n{GSM_SUFFIX}", 0.5) for r in rows]
    t0 = time.time()
    # Phase 1: room for the model prefix (2), the full thinking cap and the close token.
    gen1 = _gen(model, prompts, 2 + THINK_CAP + 1, bs)
    t1 = time.time() - t0
    final, todo, cap_hits = [None] * len(rows), [], 0
    for i, g in enumerate(gen1):
        p = _parse(g)
        if p["ans_closed"] or (g and g[-1] == EOS):
            final[i] = g
        elif p["ans_open"]:
            todo.append((i, g, ANSWER_CAP - len(p["answer"])))
        elif p["closed_think"]:  # thinking closed right at the end of phase 1: just open the answer channel
            todo.append((i, g[:g.index(END) + 1] + [MSG_MODEL, TEXT], ANSWER_CAP))
        else:  # still thinking: truncate the thinking at the cap and force the answer channel open
            k = g.index(THINK) + 1 if THINK in g else 0
            g2 = g[:k + THINK_CAP] + [END, MSG_MODEL, TEXT]
            cap_hits += 1
            todo.append((i, g2, ANSWER_CAP))
    if todo:
        conts = _gen(model, [prompts[i] + g for i, g, _ in todo], ANSWER_CAP, bs)
        for (i, g, budget), c in zip(todo, conts):
            final[i] = g + c[:max(budget, 0) + 2]  # answer budget plus its closing tokens
    per, correct, finished, think_lens = [], [], [], []
    for r, g in zip(rows, final):
        p = _parse(g)
        fin = p["ans_closed"] and len(p["answer"]) <= ANSWER_CAP
        text = tok.decode(p["answer"][:ANSWER_CAP], skip_special_tokens=True)
        pred = extract(text)
        ok = pred is not None and abs(pred - float(r["answer"])) < 1e-6
        correct.append(int(ok))
        finished.append(int(fin))
        think_lens.append(len(p["think"]))
        per.append({"id": r["id"], "pred": pred, "answer": r["answer"], "ok": ok, "finished": fin,
                    "think_tokens": len(p["think"]), "answer_tokens": len(p["answer"]), "answer_text": text[-300:]})
    c = np.array(correct, dtype=float)
    return {"metric": "gsm8k100_acc", "value": float(c.mean() * 100), "CI": boot(c), "n": len(rows),
            "finish_rate": float(np.mean(finished)), "think_cap_hits": cap_hits,
            "mean_think_tokens": float(np.mean(think_lens)), "sec": round(time.time() - t0, 1),
            "phase1_sec": round(t1, 1), "phase2_n": len(todo), "_per": per}


def main() -> None:
    import cands
    args = sys.argv[1:]
    cmd, tgt = args[0], (args[1] if len(args) > 1 and not args[1].startswith("--") else "orig")
    bs = int(args[args.index("--bs") + 1]) if "--bs" in args else 100
    model, proc = load(act_quant=AQ)
    if AQ:
        from fast_aq import enable_fast_aq
        enable_fast_aq(model)  # bit-identical to hf_load's reference W4A4 path (scripts/fidelity/test_fast_aq.py)
    cid = "orig"
    if tgt != "orig":
        cid = cands.load_candidate(model, tgt)["cand_id"]
    jobs = {"mmlu": [mmlu], "gsm8k": [gsm8k], "both": [mmlu, gsm8k]}[cmd]
    for fn in jobs:
        r = fn(model, proc) if fn is mmlu else fn(model, proc, bs)
        per = r.pop("_per")
        bench = r["metric"].split("_")[0]
        json.dump(per, open(f"{FID}/cap_{bench}_{cid}.json", "w"), indent=0)
        r.update({"cand": cid, "act_quant": AQ, "engine": "hf_w4a16" if not AQ else "hf_w4a4emu"})
        with open(f"{FID}/cap_results.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        print("RESULT", json.dumps(r), flush=True)
    cands.restore(model)


if __name__ == "__main__":
    main()
