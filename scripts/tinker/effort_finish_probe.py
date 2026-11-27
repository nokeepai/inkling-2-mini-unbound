"""How often does stock thinking close within the cap on debug tasks, by reasoning effort?

Why: RL only learns from groups with reward variance. At effort 0.9 every debug rollout hits
the 4096 thinking cap (reward 0 for all 8, group dropped), so the efforts used in training
must be ones where some rollouts finish. Stage-1 sampling only, free model.

  $PY effort_finish_probe.py --efforts 0.5,0.6,0.7,0.8 --n-prompts 10 --group 8
  $PY effort_finish_probe.py --efforts 0.9 --think-cap 16000 --n-prompts 6 --group 4 --tag long
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics

import lab_common as lc
import rollout as ro


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--efforts", default="0.5,0.6,0.7,0.8")
    ap.add_argument("--think-cap", type=int, default=4096)
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--category", default="debug")
    ap.add_argument("--tag", default="cap4096")
    args = ap.parse_args()
    rows = [r for r in lc.read_jsonl(lc.LAB_DIR / "prompts_train_code.jsonl") if r["category"] == args.category]
    rows = random.Random(7).sample(rows, args.n_prompts)
    renderer, tok = lc.renderer_and_tokenizer()
    sc = lc.sampling_client(lc.service_client())
    sem = asyncio.Semaphore(64)
    out = {}

    async def one(effort: float, row: dict) -> list[tuple[bool, int, str]]:
        cfg = ro.SampleCfg(think_cap=args.think_cap, temperature=1.0, effort=effort)
        prompt = ro.build_prompt(renderer, row["prompt"], cfg)
        async with sem:
            seqs = await ro._sample(sc, prompt, args.group, args.think_cap, [ro.END_MESSAGE], 1.0)
        return [(s.stop_reason == "length", len(s.tokens), tok.decode(list(s.tokens)[-600:])) for s in seqs]

    for e in [float(x) for x in args.efforts.split(",")]:
        res = [x for g in await asyncio.gather(*(one(e, r) for r in rows)) for x in g]
        lens = [n for _c, n, _t in res]
        fin = [n for c, n, _t in res if not c]
        out[e] = {"n": len(res), "finish_rate": round(1 - sum(c for c, _n, _t in res) / len(res), 3),
                  "median_tokens": statistics.median(lens),
                  "median_tokens_finished": statistics.median(fin) if fin else None,
                  "max_finished": max(fin) if fin else None}
        print(e, json.dumps(out[e]), flush=True)
        (lc.LAB_DIR / "logs" / f"effort_finish_{args.tag}_e{e}.tails.jsonl").write_text(
            "".join(json.dumps({"cap": c, "n": n, "tail": t}) + "\n" for c, n, t in res))
    (lc.LAB_DIR / "logs" / f"effort_finish_{args.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
