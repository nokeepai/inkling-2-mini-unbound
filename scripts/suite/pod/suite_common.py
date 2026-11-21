"""Shared helpers for the Phase 4 capability suite (agent `suite`, pod-side, /srv/suite/).

Every bench script follows one interface so suite_worker.py can run it:
  python3 bench_<name>.py gen   --port P --model M --shard k --nshards K   -> OUT/<M>/<bench>/gens_<k>of<K>.jsonl
  python3 bench_<name>.py score --model M                                 -> OUT/<M>/<bench>/score.json
score.json: {"bench", "model", "metrics": {name: {"value", "CI": [lo, hi], "n"}}, "main": <metric name>, ...}
Generation follows harness/cap/FORMAT.md (official template, no system prompt, input_ids, greedy, thinking cap +
force-close [200010, 200001, 200004]) with the suite's caps: thinking 4,096 at effort 0.9.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MSG_MODEL, TEXT, THINK, END, EOS = 200001, 200004, 200008, 200010, 200006
CONC = 31  # == --max-running-requests (ENGINE.md)
OUT = os.environ.get("SUITE_OUT", "/srv/suite/out")
DATA = os.environ.get("SUITE_DATA", "/srv/suite/data")
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("/srv/w/orig")
    return _TOK


def post(port: int, path: str, payload: dict | None, timeout: int = 3600) -> dict:
    data = json.dumps(payload).encode() if payload is not None else b"{}"
    headers = {"content-type": "application/json"}
    if path.startswith("/update_weights") and os.path.exists("/srv/admin_key"):
        headers["authorization"] = "Bearer " + open("/srv/admin_key").read().strip()
    req = urllib.request.Request(f"http://localhost:{port}{path}", data=data, headers=headers)
    for attempt in range(3):
        try:
            body = urllib.request.urlopen(req, timeout=timeout).read()
            return json.loads(body)
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(5)
    raise RuntimeError("unreachable")


def render(text: str, effort: float) -> list[int]:
    s = tok().apply_chat_template([{"role": "user", "content": text}], tokenize=False, reasoning_effort=effort)
    return tok()(s, add_special_tokens=False).input_ids


def gen(port: int, ids: list[int], n: int, stop: list[int]) -> list[int]:
    o = post(port, "/generate", {"input_ids": ids, "sampling_params": {
        "temperature": 0, "max_new_tokens": n, "stop_token_ids": stop, "skip_special_tokens": False,
        "no_stop_trim": True}})
    return o["output_ids"]


def repetition(ids: list[int], n: int = 8, k: int = 4) -> int:
    """1 if any token 8-gram occurs >= 4 times (same rule as harness/sgl_eval.py)."""
    seen: dict[tuple, int] = {}
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= k:
            return 1
    return 0


def loop_detect(ids: list[int], min_span: int = 32, times: int = 3) -> int:
    """Owner-approved true-loop detector (15:05Z), copied from harness/sgl_eval.py: a >= 32-token span repeated
    >= 3 times back to back."""
    import numpy as np
    a = np.asarray(ids, dtype=np.int64)
    n = len(a)
    for L in range(min_span, n // times + 1):
        eq = a[:-L] == a[L:]
        if eq.sum() < (times - 1) * L:
            continue
        d = np.diff(np.concatenate(([0], eq.view(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if ((ends - starts) >= (times - 1) * L).any():
            return 1
    return 0


def capped(port: int, prompt: list[int], think_cap: int, ans_cap: int) -> dict:
    """FORMAT.md generation: thinking capped at think_cap then force-closed; answer budget ans_cap."""
    out = gen(port, prompt, think_cap + 3, [END, EOS])
    think, cap_hit = [], False
    if out[:2] == [MSG_MODEL, THINK]:
        body = out[2:]
        if END in body[:think_cap + 1]:
            think = body[:body.index(END)]
            ctx = prompt + out[:2 + len(think) + 1]
            rest = gen(port, ctx, ans_cap + 2, [END, EOS])
        else:
            think, cap_hit = body[:think_cap], True
            ctx = prompt + [MSG_MODEL, THINK] + think + [END, MSG_MODEL, TEXT]
            rest = [MSG_MODEL, TEXT] + gen(port, ctx, ans_cap, [END, EOS])
    else:
        rest = out
    ans = rest[2:] if rest[:2] == [MSG_MODEL, TEXT] else rest
    finished = any(t in (END, EOS) for t in ans[:ans_cap + 1])
    ans = ans[:ans_cap]
    ans = ans[:min([ans.index(t) for t in (END, EOS) if t in ans] or [len(ans)])]
    return {"think_len": len(think), "cap_hit": cap_hit, "finished": finished, "ans_len": len(ans),
            "answer": tok().decode(ans), "rep": repetition(think + ans), "loop": loop_detect(think + ans)}


def boot(rows: list[float], seed: int = 1234, n: int = 10000) -> list[float]:
    if not rows:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    k = len(rows)
    means = sorted(sum(rows[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def stat(rows: list[float], scale: float = 1.0) -> dict:
    v = sum(rows) / max(len(rows), 1)
    lo, hi = boot(rows)
    return {"value": v * scale, "CI": [lo * scale, hi * scale], "n": len(rows)}


def hygiene(res: list[dict]) -> dict:
    return {"think_len": stat([r["think_len"] for r in res]),
            "finish_rate": stat([float(r["finished"] and not r["cap_hit"]) for r in res], 100),
            "cap_hit_rate": stat([float(r["cap_hit"]) for r in res], 100),
            "repetition_rate": stat([float(r["rep"]) for r in res], 100),
            "loop_rate": stat([float(r["loop"]) for r in res], 100)}


def shard(items: list, k: int, nshards: int) -> list:
    lim = int(os.environ.get("SUITE_LIMIT", "0"))  # smoke tests only
    return (items[:lim] if lim else items)[k::nshards]


def out_dir(model: str, bench: str) -> str:
    d = os.path.join(OUT, model, bench)
    os.makedirs(d, exist_ok=True)
    return d


def run_pool(fn, items, conc: int = CONC):
    with ThreadPoolExecutor(conc) as ex:
        return list(ex.map(fn, items))


def write_gens(model: str, bench: str, k: int, nshards: int, rows: list[dict]) -> str:
    p = os.path.join(out_dir(model, bench), f"gens_{k}of{nshards}.jsonl")
    with open(p + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(p + ".tmp", p)
    return p


def read_gens(model: str, bench: str) -> list[dict]:
    d = out_dir(model, bench)
    files = sorted(f for f in os.listdir(d) if f.startswith("gens_") and f.endswith(".jsonl"))
    ks = {f.split("of")[1].split(".")[0] for f in files}
    assert len(ks) == 1, f"mixed shard counts in {d}: {files}"
    nsh = int(ks.pop())
    assert len(files) == nsh, f"{d}: {len(files)} of {nsh} shards present"
    rows = [json.loads(l) for f in files for l in open(os.path.join(d, f))]
    return sorted(rows, key=lambda r: r["idx"])


def write_score(model: str, bench: str, metrics: dict, main: str, extra: dict | None = None) -> dict:
    s = {"bench": bench, "model": model, "main": main, "metrics": metrics, **(extra or {})}
    json.dump(s, open(os.path.join(out_dir(model, bench), "score.json"), "w"), indent=1)
    print(json.dumps({k: round(v["value"], 3) for k, v in metrics.items()}))
    return s


def cli(bench: str, gen_fn, score_fn) -> None:
    """gen_fn(port, model, k, nshards) -> rows (each with "idx"); score_fn(model) -> None (writes score.json)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "score"])
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "gen":
        t = time.time()
        rows = gen_fn(a.port, a.model, a.shard, a.nshards)
        p = write_gens(a.model, bench, a.shard, a.nshards, rows)
        print(f"wrote {len(rows)} rows to {p} in {time.time() - t:.0f}s")
    else:
        score_fn(a.model)
