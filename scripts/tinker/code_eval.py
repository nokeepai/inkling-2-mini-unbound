"""Small code eval: 40 MBPP (sanitized, test split) problems, fixed seed, never in training.

The model's last ```python block runs with the problem's asserts in a fresh subprocess
(python -I, temp dir, 10 s timeout). Reports pass@1 (one sample, t0.6), median thinking
tokens and thinking-cap-hit rate.

  PY=~/.inkling-abl/tinker_lab/.venv/bin/python
  $PY scripts/tinker/code_eval.py --tag base_code [--model-path tinker://...] [--effort 0.9]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import lab_common as lc
import rollout as ro

SEED = 31337
N = 40
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def problems() -> list[dict[str, Any]]:
    path = lc.LAB_DIR / "prompts_code.jsonl"
    if path.exists():
        return lc.read_jsonl(path)
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    rows = []
    for i in random.Random(SEED).sample(range(len(ds)), N):
        r = ds[i]
        prompt = (f"{r['prompt']}\n\nYour function must pass this test:\n{r['test_list'][0]}\n\n"
                  "Reply with the complete Python function in a single ```python code block.")
        rows.append({"id": f"mbpp{r['task_id']}", "category": "code", "prompt": prompt,
                     "tests": r["test_imports"] + r["test_list"], "target_words": None, "gold": None})
    lc.write_jsonl(path, rows)
    return rows


def run_tests(answer: str, tests: list[str]) -> tuple[bool, str]:
    blocks = CODE_RE.findall(answer)
    if not blocks:
        return False, "no code block"
    program = blocks[-1] + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "solution.py"
        f.write_text(program)
        try:
            res = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True,
                                 text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return res.returncode == 0, res.stderr[-300:]


async def run(args: argparse.Namespace) -> None:
    cfg = ro.SampleCfg(temperature=args.temperature, effort=args.effort, force_answer_after_cap=True)
    renderer, tok = lc.renderer_and_tokenizer()
    svc = lc.service_client()
    if args.model_path:
        sc = await svc.create_sampling_client(model_path=args.model_path)
        lc.assert_free(await sc.get_base_model_async())
    else:
        sc = await lc.sampling_client(svc)
    sem = asyncio.Semaphore(32)
    rows = problems()
    t0 = time.time()

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        r = (await ro.rollout_group(sc, ro.build_prompt(renderer, row["prompt"], cfg), 1, cfg, sem))[0]
        s = ro.to_sample(r, tok, row, cfg)
        ok, err = await asyncio.to_thread(run_tests, s.answer, row["tests"])
        return {"id": row["id"], "passed": ok, "err": err, "think_tokens": s.think_tokens,
                "answer_tokens": s.answer_tokens, "cap_hit": s.think_cap_hit,
                "runaway": s.answer_runaway, "answer": s.answer}

    recs = await asyncio.gather(*(one(r) for r in rows))
    lc.write_jsonl(lc.LAB_DIR / "eval" / f"{args.tag}.jsonl", list(recs))
    n = len(recs)
    summary = {"tag": args.tag, "model_path": args.model_path, "effort": args.effort,
               "temperature": args.temperature, "n": n,
               "pass_at_1": sum(r["passed"] for r in recs) / n,
               "median_think_tokens": statistics.median(r["think_tokens"] for r in recs),
               "cap_hit": sum(r["cap_hit"] for r in recs) / n,
               "runaway": sum(r["runaway"] for r in recs) / n, "wall_s": round(time.time() - t0)}
    (lc.LAB_DIR / "eval" / f"{args.tag}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--effort", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.6)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
