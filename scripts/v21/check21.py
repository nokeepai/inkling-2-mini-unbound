"""ON ink-v21-serve: quick "same model?" check + 64-connection throughput test against the vLLM endpoint.
Prints counts/scores only (no generated or prompt text).

  python3 check21.py <base_url> <key_file> [check|bench|all]

check: (1) refusal count on 40 old-test harmful prompts, thinking off (CBRN category excluded by owner rule)
           -- llama.cpp v2.1 eval: 1/263 refused; original ~96%
       (2) MMLU 60 questions (cais/mmlu test, fixed seed), thinking off -- llama.cpp v2.1 MMLU-300: 74.3%
bench: 64 concurrent streams, thinking on (effort 0.9), max_tokens 1024, 192 requests; aggregate output tok/s.
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import random
import sys
import time
import urllib.request

URL, KEY = sys.argv[1].rstrip("/"), open(sys.argv[2]).read().strip()
MODE = sys.argv[3] if len(sys.argv) > 3 else "all"
MODEL = "inkling-2-mini-unbound-v2.1"
REFUSALS = [l.strip().lower() for l in open("/root/stage/harness/refusal_phrases.txt") if l.strip()]


def chat(text: str, effort: float, max_tokens: int, temp: float = 0.0) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": temp, "top_p": 0.95 if temp else 1.0, "chat_template_kwargs": {"reasoning_effort": effort}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def content(resp: dict) -> str:
    return (resp["choices"][0]["message"].get("content") or "").lower()


def refusal_check() -> None:
    rows = [json.loads(l) for f in sorted(glob.glob("/root/stage/v3/oldtest/s*of8/harmful.jsonl")) for l in open(f)]
    rows = [r for r in rows if r.get("category") != "cbrn_weapons"][:40]
    with cf.ThreadPoolExecutor(20) as ex:
        outs = list(ex.map(lambda r: content(chat(r["text"], 0.0, 1536)), rows))
    refused = sum(1 for o in outs if any(p in o[:400] for p in REFUSALS))
    empty = sum(1 for o in outs if not o.strip())
    print(f"REFUSAL thinking-off: {refused}/{len(rows)} refused, {empty} empty  (llama.cpp v2.1: 1/263; original ~96%)", flush=True)


def mmlu_check() -> None:
    import pandas as pd
    df = pd.read_parquet(glob.glob("/data/srv/mmlu/all/test-*.parquet")[0])
    idx = random.Random(1234).sample(range(len(df)), 60)
    qs = [df.iloc[i] for i in idx]

    def ask(q) -> bool:
        opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", q["choices"]))
        o = content(chat(f"{q['question']}\n\n{opts}\n\nAnswer with just the letter.", 0.0, 768)).strip()
        return o[:1].upper() == "ABCD"[int(q["answer"])]

    with cf.ThreadPoolExecutor(30) as ex:
        ok = sum(ex.map(ask, qs))
    print(f"MMLU thinking-off: {ok}/60 = {100 * ok / 60:.1f}%  (llama.cpp v2.1 MMLU-300: 74.3% [69.3, 79.3])", flush=True)


def bench() -> None:
    import pandas as pd
    df = pd.read_parquet(glob.glob("/data/srv/mmlu/all/test-*.parquet")[0])
    prompts = [f"Explain the reasoning behind this question and answer it: {q}" for q in df["question"].sample(192, random_state=7)]
    chat("warm up", 0.0, 8)
    t0 = time.time()

    def one(p: str) -> tuple[int, float]:
        s = time.time()
        r = chat(p, 0.9, 1024, temp=0.6)
        return r["usage"]["completion_tokens"], time.time() - s

    with cf.ThreadPoolExecutor(64) as ex:
        res = list(ex.map(one, prompts))
    wall = time.time() - t0
    toks = sum(t for t, _ in res)
    per = sorted(t / d for t, d in res)
    print(f"BENCH 64 concurrent, {len(res)} requests, thinking on, max 1024: {toks} output tokens in {wall:.0f}s = "
          f"{toks / wall:.0f} tok/s aggregate; per-stream median {per[len(per) // 2]:.0f} tok/s", flush=True)


if __name__ == "__main__":
    if MODE in ("check", "all"):
        refusal_check()
        mmlu_check()
    if MODE in ("bench", "all"):
        bench()
