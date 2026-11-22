"""ON eval pod: continue samples whose ANSWER hit the old --ans-cap (finished=False, cap_hit=False) up to a new cap.

  python3 cont_samples.py --ports 8106,8107 --inp gens/X.jsonl --out gens/X.cont.jsonl --ans-cap 8192 [--greedy]

Prompt = the same rendered prompt (system prompt if sys=1, same effort) + the row's completion tokens so far; the model
continues the answer with the same sampling settings (new seed). Equivalent to having sampled with the larger cap
(a sample from the model given the same prefix), at a fraction of the cost of resampling. Rows are rewritten with the
extended answer/completion/finished; resumable by (id, k).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_samples import END, EOS, TEXT, MSG_MODEL, TOK, gen, render  # noqa: E402

LOCK = threading.Lock()


def answer_start(comp: list[int]) -> int:
    """Index of the first answer token: after the last [MSG_MODEL, TEXT] marker."""
    for i in range(len(comp) - 2, -1, -1):
        if comp[i] == MSG_MODEL and comp[i + 1] == TEXT:
            return i + 2
    raise ValueError("no answer marker")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ans-cap", type=int, default=8192)
    ap.add_argument("--system", default="sys_uncensored.txt")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--conc", type=int, default=16)
    a = ap.parse_args()
    system = open(a.system).read().strip()
    ports = [int(p) for p in a.ports.split(",")]
    rows = [json.loads(l) for l in open(a.inp)]
    todo = [r for r in rows if not r["finished"] and not r["cap_hit"]]
    done = set()
    if os.path.exists(a.out):
        done = {(r["id"], r["k"]) for r in map(json.loads, open(a.out))}
    todo = [r for r in todo if (r["id"], r["k"]) not in done]
    print(f"{len(todo)} to continue ({len(rows)} rows)", flush=True)
    f = open(a.out, "a")
    cnt = [0]

    def one(ix_r):
        ix, r = ix_r
        port = ports[ix % len(ports)]
        comp = list(r["completion"])
        s = answer_start(comp)
        ans = comp[s:]
        samp = ({"temperature": 0.0, "top_k": 1, "seed": 1234} if a.greedy else
                {"temperature": a.temp, "top_p": a.top_p, "top_k": 0,
                 "seed": 2000 + 7919 * r["k"] + zlib.crc32(r["id"].encode()) % 100000})
        prompt = render(r["prompt"], float(r["effort"]), system if r.get("sys") else None)
        try:
            more = gen(port, prompt + comp, max(1, a.ans_cap - len(ans)), samp)
        except Exception as e:  # retried on resume
            print("ERR", r["id"], r["k"], repr(e)[:200], flush=True)
            return
        stop = min([more.index(t) for t in (END, EOS) if t in more] or [len(more)])
        body = ans + more[:stop]
        finished = stop < len(more) and len(body) <= a.ans_cap
        r2 = dict(r, answer=TOK.decode(body), finished=finished, completion=comp[:s] + body + ([more[stop]] if finished else []),
                  ans_len=len(body), continued=1)
        with LOCK:
            f.write(json.dumps(r2) + "\n")
            f.flush()
            cnt[0] += 1
            if cnt[0] % 200 == 0:
                print("done", cnt[0], flush=True)

    with ThreadPoolExecutor(a.conc * len(ports)) as ex:
        list(ex.map(one, enumerate(todo)))
    print("CONT_DONE", a.out, cnt[0], flush=True)


if __name__ == "__main__":
    main()
