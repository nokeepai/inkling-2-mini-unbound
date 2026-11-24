"""ON the training pod: the student's mean per-token NLL of each row's answer under thinking-off framing.

  python3 score_nll.py --data rows.jsonl --out nll.jsonl [--base /root/w/v10] [--lora run/lora_final.pt]
      [--effort 0] [--max-len 6144] [--limit N]

Rows: {"id", "k"?, "text" (user prompt) or "messages" (full prompt list, e.g. rp_play rows), "system"?,
       one of: "completion" (token ids; the answer = tokens after the last <|message_model|><|content_text|> up to
       <|end_message|>/eos), "completion_text", or "answer" (text, tokenized here)}.
Sequence = encode(prompt, effort=--effort, system) + [MSG_MODEL, TEXT] + answer + [END]. NLL is averaged over the answer
tokens and the closing END (the forced header is context, not scored). Thinking-off framing on purpose: it measures how
natural the answer itself is for the student, independent of whatever thinking produced it.
--lora merges a train_lora.py adapter (W += alpha/r * B A) into the bf16 weights before scoring.
Out rows: {id, k, turn, nll, n_tok, truncated}. Batch size 1, no grad. Resumable on (id, k, turn).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "/root/parity")
from hf_load import encode, load  # noqa: E402

MSG_MODEL, TEXT, END, EOS = 200001, 200004, 200010, 200006


def answer_ids(row: dict, tok) -> list[int]:
    if row.get("completion"):
        c = list(row["completion"])
        starts = [i for i in range(len(c) - 1) if c[i] == MSG_MODEL and c[i + 1] == TEXT]
        body = c[starts[-1] + 2:] if starts else c
        stops = [i for i, t in enumerate(body) if t in (END, EOS)]
        return body[:stops[0]] if stops else body
    text = row.get("completion_text") or row.get("answer") or ""
    return tok(text, add_special_tokens=False).input_ids


def merge_lora(model, path: str) -> None:
    blob = torch.load(path, map_location="cpu")
    args = blob["meta"]["args"]
    scale = float(args["alpha"]) / int(args["rank"])
    for name, (A, B) in blob["lora"].items():
        lin = model.get_submodule(name)
        w = lin.weight
        w.data += (scale * (B.float() @ A.float())).to(device=w.device, dtype=w.dtype)
    print("merged", len(blob["lora"]), "lora modules, scale", scale, flush=True)


@torch.no_grad()
def nll_of(model, prompt: list[int], ans: list[int]) -> float:
    target = ans + [END]
    ids = torch.tensor([prompt + [MSG_MODEL, TEXT] + target], device="cuda:0")
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    start = len(prompt) + 2
    lp = logits[0, start - 1: start - 1 + len(target)].float()
    return torch.nn.functional.cross_entropy(lp, torch.tensor(target, device=lp.device), reduction="mean").item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="/root/w/v10")
    ap.add_argument("--lora", default="")
    ap.add_argument("--effort", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    model, proc = load(a.base)
    model.requires_grad_(False)
    model.eval()
    if a.lora:
        merge_lora(model, a.lora)
    tok = getattr(proc, "tokenizer", proc)
    done = set()
    if os.path.exists(a.out):
        done = {(r["id"], r.get("k", 0), r.get("turn", 0)) for r in map(json.loads, open(a.out))}
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    todo = [r for r in rows if (r["id"], r.get("k", 0), r.get("turn", 0)) not in done][: a.limit or None]
    print(f"{len(todo)} rows to score", flush=True)
    f = open(a.out, "a")
    for i, r in enumerate(todo):
        user = r.get("messages") or r["text"]
        prompt = encode(proc, user, effort=a.effort, system=None if r.get("messages") else r.get("system"))[0].tolist()
        ans = answer_ids(r, tok)
        trunc = len(prompt) + len(ans) + 3 > a.max_len
        if trunc:
            ans = ans[: max(1, a.max_len - len(prompt) - 3)]
        nll = nll_of(model, prompt, ans)
        f.write(json.dumps({"id": r["id"], "k": r.get("k", 0), "turn": r.get("turn", 0), "nll": round(nll, 5), "n_tok": len(ans) + 1,
                            "truncated": trunc}) + "\n")
        f.flush()
        if (i + 1) % 100 == 0:
            print("scored", i + 1, flush=True)
    print("NLL_DONE", a.out, len(todo), flush=True)


if __name__ == "__main__":
    main()
