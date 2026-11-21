"""RULER at 32k and 128k (agent suite/ruler). Pod-side, /srv/suite/ruler/bench_ruler.py, suite_common interface:
  python3 bench_ruler.py gen   --port P --model M [--shard k --nshards K]
  python3 bench_ruler.py score --model M
Data: /srv/suite/data/ruler/<task>_<len>.jsonl (gen_ruler_data.py, lm-eval 0.4.9.1 generators, seed 1234).
Prompt: official chat template at reasoning_effort=0 (thinking off), user turn = RULER input, then
[200001, 200004] (<|message_model|><|content_text|>) + tokens of the task's answer prefix. Greedy.
max_new_tokens per RULER task: niah 128, vt 30, cwe 120, fwe 50, qa 32.
Scoring (NVIDIA RULER): string_match_all for niah/vt/cwe/fwe (fraction of refs found), string_match_part for qa
(1 if any ref found). Prediction postprocess as RULER (strip, control chars -> newline).
Env RULER_N128 caps samples per task at 128k (first k by index); RULER_N32 likewise at 32k.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time

sys.path.insert(0, "/srv/suite")
import suite_common as sc  # noqa: E402

BENCH = "ruler"
DATA = os.path.join(sc.DATA, "ruler")
TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]
LENGTHS = {32768: "ruler_32k", 131072: "ruler_128k"}
MAX_NEW = {"vt": 30, "cwe": 120, "fwe": 50, "qa_1": 32, "qa_2": 32}  # niah_*: 128
CONC = int(os.environ.get("RULER_CONC", "16"))  # == --max-running-requests in ENGINE_LC.md
TOK_LOCK = threading.Lock()  # HF fast tokenizer is not safe across threads


def max_new(task: str) -> int:
    return MAX_NEW.get(task, 128)


def caps() -> dict:
    return {32768: int(os.environ.get("RULER_N32", "25")), 131072: int(os.environ.get("RULER_N128", "10"))}


def items() -> list[dict]:
    out, c = [], caps()
    for seq in LENGTHS:
        for t in TASKS:
            rows = [json.loads(l) for l in open(f"{DATA}/{t}_{seq}.jsonl")]
            rows = sorted(rows, key=lambda r: r["index"])[:c[seq]]
            for r in rows:
                out.append({"task": t, "seq": seq, "sidx": r["index"], "input": r["input"], "outputs": r["outputs"],
                            "gen_prefix": r["gen_prefix"]})
    for i, r in enumerate(out):
        r["idx"] = i
    return out


def prompt_ids(r: dict) -> list[int]:
    ids = sc.render(r["input"], 0) + [sc.MSG_MODEL, sc.TEXT]
    if r["gen_prefix"]:
        ids += sc.tok()(r["gen_prefix"], add_special_tokens=False).input_ids
    return ids


def one(port: int, r: dict) -> dict:
    with TOK_LOCK:
        ids = prompt_ids(r)
    t = time.time()
    out = sc.gen(port, ids, max_new(r["task"]), [sc.END, sc.EOS])
    stop = min([out.index(x) for x in (sc.END, sc.EOS) if x in out] or [len(out)])
    with TOK_LOCK:
        pred = sc.tok().decode(out[:stop])
    return {"idx": r["idx"], "task": r["task"], "seq": r["seq"], "sidx": r["sidx"], "prompt_len": len(ids),
            "out_len": len(out), "finished": stop < len(out), "pred": pred,
            "outputs": r["outputs"], "secs": round(time.time() - t, 2)}


def gen_fn(port: int, model: str, k: int, nshards: int) -> list[dict]:
    its = sc.shard(items(), k, nshards)
    # longest first so the 128k prompts do not straggle at the end
    its = sorted(its, key=lambda r: -r["seq"])
    done = [0]
    t0 = time.time()

    def f(r):
        res = one(port, r)
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"{done[0]}/{len(its)} {time.time() - t0:.0f}s", flush=True)
        return res
    return sorted(sc.run_pool(f, its, CONC), key=lambda r: r["idx"])


def post(pred: str) -> str:
    return re.sub(r"[\x00-\x1f]", "\n", pred.strip()).strip()


def sample_score(r: dict) -> float:
    p = post(r["pred"]).lower()
    hits = [1.0 if x.lower() in p else 0.0 for x in r["outputs"]]
    if r["task"].startswith("qa"):
        return max(hits)  # string_match_part
    return sum(hits) / len(hits)  # string_match_all


def strat_boot(by_task: dict[str, list[float]], seed: int = 1234, n: int = 10000) -> list[float]:
    """Bootstrap over samples, stratified by task; statistic = mean over tasks of task accuracy."""
    rng = random.Random(seed)
    tasks = sorted(by_task)
    stats = []
    for _ in range(n):
        tot = 0.0
        for t in tasks:
            xs = by_task[t]
            k = len(xs)
            tot += sum(xs[rng.randrange(k)] for _ in range(k)) / k
        stats.append(tot / len(tasks))
    stats.sort()
    return [stats[int(0.025 * n)], stats[int(0.975 * n) - 1]]


def score_fn(model: str) -> None:
    c = caps()
    rows = [r for r in sc.read_gens(model, BENCH) if r["sidx"] < c[r["seq"]]]  # same fixed subset for every model
    metrics, per_task, hyg = {}, {}, {}
    for seq, name in LENGTHS.items():
        by_task = {}
        for t in TASKS:
            xs = [sample_score(r) for r in rows if r["seq"] == seq and r["task"] == t]
            if xs:
                by_task[t] = xs
        if not by_task:
            continue
        v = sum(sum(x) / len(x) for x in by_task.values()) / len(by_task)
        lo, hi = strat_boot(by_task)
        n = sum(len(x) for x in by_task.values())
        metrics[name] = {"value": 100 * v, "CI": [100 * lo, 100 * hi], "n": n, "n_tasks": len(by_task)}
        for t, xs in by_task.items():
            s = sc.stat(xs, 100)
            metrics[f"{name}/{t}"] = s
        sub = [r for r in rows if r["seq"] == seq]
        hyg[name] = {"finish_rate": 100 * sum(r["finished"] for r in sub) / len(sub),
                     "mean_out_len": sum(r["out_len"] for r in sub) / len(sub),
                     "max_prompt_len": max(r["prompt_len"] for r in sub),
                     "gen_secs_sum": round(sum(r["secs"] for r in sub), 1)}
    sc.write_score(model, BENCH, metrics, "ruler_128k" if "ruler_128k" in metrics else "ruler_32k",
                   {"hygiene": hyg, "caps": caps()})


if __name__ == "__main__":
    sc.cli(BENCH, gen_fn, score_fn)
