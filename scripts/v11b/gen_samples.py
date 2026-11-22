"""ON eval pod: sample completions from llama-servers for SFT data (v1.1 rejection-sampling distillation).

  python3 gen_samples.py --ports 8100,8101 --inp prompts.jsonl --out out.jsonl --effort 0 --n 4 --temp 0.8 \
      [--system sys.txt] [--think-cap 1536] [--ans-cap 768] [--conc 12] [--greedy]

Each output row: id, prompt, effort, sys (0/1), k (sample index), think_len, cap_hit, finished, answer, think,
completion (token ids from the model's first token through <|end_message|>, exactly what the model produced; the
force-close path is used only on cap hits, which are flagged and never used for training).
Prompt ids are rendered with the official template (HF tokenizer), with the system prompt only if --system is given.
Resumable: rows already in --out (id, k) are skipped.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import threading
import zlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

MSG_MODEL, TEXT, THINK, END, EOS = 200001, 200004, 200008, 200010, 200006
TOK = AutoTokenizer.from_pretrained(os.environ.get("CMP_TOK", "/data/compare/models/tok"))
END_S, EOS_S = TOK.decode([END]), TOK.decode([EOS])
LOCK = threading.Lock()


def post(port: int, payload: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def render(text: str, effort: float, system: str | None) -> list[int]:
    msgs = [{"role": "user", "content": text}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    s = TOK.apply_chat_template(msgs, tokenize=False, reasoning_effort=effort)
    return TOK(s, add_special_tokens=False).input_ids


def gen(port: int, ids: list[int], n: int, samp: dict) -> list[int]:
    o = post(port, {"prompt": ids, "n_predict": n, "cache_prompt": True, "return_tokens": True,
                    "stop": [END_S, EOS_S], **samp})
    toks = list(o.get("tokens") or [])
    st = o.get("stop_type")
    if st == "eos" and (not toks or toks[-1] not in (END, EOS)):
        toks.append(EOS)
    elif st == "word" and (not toks or toks[-1] not in (END, EOS)):
        toks.append(END if o.get("stopping_word") == END_S else EOS)
    return toks


def sample(port: int, prompt: list[int], think_cap: int, ans_cap: int, samp: dict) -> dict:
    out = gen(port, prompt, think_cap + 3, samp)
    think, cap_hit, comp = [], False, []
    if out[:2] == [MSG_MODEL, THINK]:
        body = out[2:]
        if END in body[:think_cap + 1]:
            think = body[:body.index(END)]
            comp = out[:2 + len(think) + 1]
            rest = gen(port, prompt + comp, ans_cap + 2, samp)
        else:
            think, cap_hit = body[:think_cap], True
            comp = [MSG_MODEL, THINK] + think + [END, MSG_MODEL, TEXT]
            rest = [MSG_MODEL, TEXT] + gen(port, prompt + comp, ans_cap, samp)
            comp = comp[:-2]
    else:
        rest = out
    ans = rest[2:] if rest[:2] == [MSG_MODEL, TEXT] else rest
    stop = min([ans.index(t) for t in (END, EOS) if t in ans] or [len(ans)])
    finished = stop < len(ans) and stop <= ans_cap
    ans_body = ans[:min(stop, ans_cap)]
    comp = comp + rest[:2] + ans_body + ([ans[stop]] if finished else [])
    return {"think_len": len(think), "cap_hit": cap_hit, "finished": finished, "answer": TOK.decode(ans_body),
            "think": TOK.decode(think) if think else "", "completion": comp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--system", default="")
    ap.add_argument("--think-cap", type=int, default=1536)
    ap.add_argument("--ans-cap", type=int, default=768)
    ap.add_argument("--conc", type=int, default=12, help="concurrent requests per port")
    ap.add_argument("--seed0", type=int, default=1000)
    a = ap.parse_args()
    ports = [int(p) for p in a.ports.split(",")]
    system = open(a.system).read().strip() if a.system else None
    rows = [json.loads(l) for l in open(a.inp)]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done.add((r["id"], r["k"]))
    jobs = [(r, k) for r in rows for k in range(a.n) if (r["id"], k) not in done]
    print(f"{len(jobs)} jobs over {len(ports)} ports", flush=True)
    port_cycle = itertools.cycle(ports)
    f = open(a.out, "a")
    n_done = [0]

    def one(job):
        r, k = job
        with LOCK:
            port = next(port_cycle)
        samp = ({"temperature": 0.0, "top_k": 1, "seed": 1234} if a.greedy else
                {"temperature": a.temp, "top_p": a.top_p, "top_k": 0, "seed": a.seed0 + 7919 * k + zlib.crc32(r["id"].encode()) % 100000})
        try:
            s = sample(port, render(r["text"], a.effort, system), a.think_cap, a.ans_cap, samp)
        except Exception as e:  # keep going; the row is retried on resume
            print("ERR", r["id"], k, repr(e)[:200], flush=True)
            return
        s.update({"id": r["id"], "prompt": r["text"], "category": r.get("category"), "effort": a.effort,
                  "sys": int(bool(system)), "k": k})
        with LOCK:
            f.write(json.dumps(s) + "\n")
            f.flush()
            n_done[0] += 1
            if n_done[0] % 200 == 0:
                print("done", n_done[0], flush=True)

    with ThreadPoolExecutor(a.conc * len(ports)) as ex:
        list(ex.map(one, jobs))
    print("GEN_DONE", a.out, n_done[0], flush=True)


if __name__ == "__main__":
    main()
