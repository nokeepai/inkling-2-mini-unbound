"""Mac: teacher answers from Venice's uncensored DeepSeek for a prompts jsonl (answers only; reasoning is discarded).

  python3 teacher_venice.py in.jsonl out.jsonl [--model deepseek-v4-1-flash-uncensored] [--max-tokens 6000]
      [--conc 8] [--max-usd 3] [--system sys.txt] [--no-think] [--limit N]

Input rows: {"id", "text", "system"? (overrides --system), "category"?}. Venice system prompt is disabled
(venice_parameters.include_venice_system_prompt = false). Model id checked 2026-11-23 at /api/v1/models:
"deepseek-v4-1-flash-uncensored" (DeepSeek V4.1 Flash Uncensored, $0.33 in / $1.32 out per 1M tokens; reasoning tokens
bill as output). Output rows: id, prompt, category, k (0), teacher model, answer, finish_reason, finished, usage, cost, error.
Cost = from usage at the listed prices; --max-usd stops new calls (running total printed). Resumable on id (errors retried).
Key: uncensoredbench tools/pilot.py key("venice"); never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.expanduser("~/Code/uncensoredbench/tools"))
import pilot  # noqa: E402

URL = "https://api.venice.ai/api/v1/chat/completions"
PRICES = {"deepseek-v4-1-flash-uncensored": (0.33, 1.32), "deepseek-v4-1-flash-1017-uncensored": (0.33, 1.32),
          "venice-uncensored-1-2": (0.2, 0.9)}
THINK_RX = re.compile(r"<think>.*?</think>\s*", re.S)


def ask(key: str, model: str, msgs: list[dict], max_tokens: int, no_think: bool) -> tuple[dict | None, str | None]:
    vp = {"include_venice_system_prompt": False, "strip_thinking_response": True}
    if no_think:
        vp["disable_thinking"] = True
    body = json.dumps({"model": model, "messages": msgs, "max_tokens": max_tokens, "venice_parameters": vp}).encode()
    err = "unknown"
    for attempt in range(6):
        req = urllib.request.Request(URL, body, {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=900)), None
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:200].decode('utf8', 'replace')}".replace(key, "<key>")
            if e.code not in (408, 429, 500, 502, 503, 504, 529):
                return None, err
        except Exception as e:  # noqa: BLE001 - network errors retried, then recorded
            err = f"{type(e).__name__}: {str(e)[:200]}".replace(key, "<key>")
        time.sleep(min(60, 10 * (attempt + 1)))
    return None, err


def cost_of(model: str, usage: dict) -> float:
    pin, pout = PRICES.get(model, (0.33, 1.32))
    return (int(usage.get("prompt_tokens") or 0) * pin + int(usage.get("completion_tokens") or 0) * pout) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--model", default="deepseek-v4-1-flash-uncensored")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-usd", type=float, default=3.0)
    ap.add_argument("--system", default="")
    ap.add_argument("--no-think", action="store_true", help="ask Venice to disable the model's reasoning")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    key = pilot.key("venice")
    system = open(a.system).read().strip() if a.system else None
    done, spent = set(), [0.0]
    if os.path.exists(a.out):
        for r in map(json.loads, open(a.out)):
            spent[0] += r.get("cost") or 0.0  # earlier runs count toward the cap too
            if r.get("error") is None:
                done.add(r["id"])
    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    todo = [r for r in rows if r["id"] not in done][: a.limit or None]
    print(f"{len(todo)} to ask {a.model} ({len(done)} done, ${spent[0]:.3f} spent so far)", flush=True)
    lock, f, n = threading.Lock(), open(a.out, "a"), [0, 0]

    def one(r: dict) -> None:
        if spent[0] >= a.max_usd:
            n[1] += 1
            return
        rsys = r.get("system", system)
        msgs = ([{"role": "system", "content": rsys}] if rsys else []) + [{"role": "user", "content": r["text"]}]
        resp, err = ask(key, a.model, msgs, a.max_tokens, a.no_think)
        row = {"id": r["id"], "prompt": r["text"], "category": r.get("category"), "k": 0, "teacher": a.model,
               "sys": int(bool(rsys)), "effort": 0.0, "error": err}
        if resp is not None:
            ch = (resp.get("choices") or [{}])[0]
            ans = THINK_RX.sub("", (ch.get("message") or {}).get("content") or "").strip()
            usage = resp.get("usage") or {}
            row.update({"answer": ans, "finish_reason": ch.get("finish_reason"),
                        "finished": ch.get("finish_reason") == "stop", "usage": usage, "cost": cost_of(a.model, usage)})
        with lock:
            spent[0] += row.get("cost") or 0.0
            f.write(json.dumps(row) + "\n")
            f.flush()
            n[0] += 1
            if n[0] % 25 == 0:
                print(f"{n[0]}/{len(todo)} spent ${spent[0]:.3f}", flush=True)

    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(one, todo))
    if n[1]:
        print(f"STOPPED at --max-usd {a.max_usd}: {n[1]} prompts not asked", flush=True)
    print(f"TEACHER_DONE {a.out} {n[0]} rows, total spent ${spent[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
