"""Speed (agent `suite`): fixed 200-prompt workload = 200 IFEval prompts sampled with seed 1234, effort 0, greedy,
max_new_tokens 512 (stop at eos), raw /generate (no thinking cap logic).

  python3 bench_speed.py speed gen --port P --model M       (bench name "speed" or "speed_dspark")
  python3 bench_speed.py speed score --model M
Measures: aggregate output tokens/s at concurrency 31 (all 200), single-stream tokens/s (first 10 prompts, one at a time).
For a DSpark server, acceptance length = completion_tokens / spec_verify_ct from meta_info (when the server reports it).
GPU-type pinned (RTX 5090) and GPU-index pinned per job (worker field "gpu").
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suite_common as S  # noqa: E402


def prompts() -> list[list[int]]:
    rows = [json.loads(l) for l in open(f"{S.DATA}/ifeval.jsonl")]
    pick = sorted(random.Random(1234).sample(range(len(rows)), 200))
    return [S.render(rows[i]["prompt"], 0) for i in pick]


def one(port: int, ids: list[int]) -> dict:
    t = time.time()
    o = S.post(port, "/generate", {"input_ids": ids, "sampling_params": {
        "temperature": 0, "max_new_tokens": 512, "stop_token_ids": [S.EOS], "skip_special_tokens": False}})
    m = o.get("meta_info", {})
    return {"sec": time.time() - t, "out": m.get("completion_tokens", len(o.get("output_ids", []))),
            "verify": m.get("spec_verify_ct"), "accept": m.get("spec_accept_length")}


def gen(port: int, model: str, k: int, nsh: int) -> list[dict]:
    P = prompts()
    one(port, P[0])  # warm-up
    single = []
    for i, ids in enumerate(P[:10]):
        r = one(port, ids)
        single.append({"idx": i, "mode": "single", **r})
    t = time.time()
    conc = S.run_pool(lambda ids: one(port, ids), P)
    wall = time.time() - t
    rows = single + [{"idx": 100 + i, "mode": "conc31", **r} for i, r in enumerate(conc)]
    rows.append({"idx": 10_000, "mode": "wall", "sec": wall, "out": sum(r["out"] for r in conc)})
    return rows


def score(model: str, bench: str) -> None:
    rows = S.read_gens(model, bench)
    wall = next(r for r in rows if r["mode"] == "wall")
    single = [r for r in rows if r["mode"] == "single"]
    conc = [r for r in rows if r["mode"] == "conc31"]
    met = {"tok_s_conc31": {"value": wall["out"] / wall["sec"], "CI": None, "n": len(conc)},
           "tok_s_single": S.stat([r["out"] / r["sec"] for r in single]),
           "mean_out_tokens": S.stat([r["out"] for r in conc])}
    ver = [r for r in conc if r.get("verify")]
    if ver:
        met["dspark_accept_len"] = S.stat([r["out"] / r["verify"] for r in ver])
    S.write_score(model, bench, {f"{bench}_{k}": v for k, v in met.items()}, f"{bench}_tok_s_conc31")


if __name__ == "__main__":
    bench = sys.argv.pop(1)
    S.cli(bench, gen, lambda m: score(m, bench))
