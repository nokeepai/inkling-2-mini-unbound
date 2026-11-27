"""Training pool with a ~25% hard slice: prompts_train.jsonl + 70 MATH train level 4-5
(integer answers) + 30 generated Python code-output problems (answers computed here).
MATH comes from the train split, so it is disjoint from prompts_hardmath.jsonl (test splits).

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/hardpool.py
"""
from __future__ import annotations

import contextlib
import io
import random
import re
from typing import Any

import lab_common as lc
from prompts import BLOCKLIST, _last_boxed

SEED = 777
SUFFIX = "\n\nGive the final answer in \\boxed{}."


def math_train_hard(n: int, rng: random.Random) -> list[dict[str, Any]]:
    from datasets import load_dataset

    pool = []
    for s in ["algebra", "prealgebra", "number_theory", "counting_and_probability",
              "intermediate_algebra", "geometry", "precalculus"]:
        for r in load_dataset("EleutherAI/hendrycks_math", s, split="train"):
            gold = _last_boxed(r["solution"])
            if r["level"] in ("Level 4", "Level 5") and gold and re.fullmatch(r"-?\d+", gold) \
                    and not BLOCKLIST.search(r["problem"]):
                pool.append({"category": "maths", "source": f"math_train_hard_{s}",
                             "prompt": r["problem"] + SUFFIX, "target_words": None, "gold": gold})
    return rng.sample(pool, n)


def _program(rng: random.Random) -> str:
    """Small deterministic programs with an integer final print; templates vary structure."""
    a, b, c, n = rng.randint(2, 9), rng.randint(2, 7), rng.randint(1, 5), rng.randint(5, 12)
    templates = [
        f"total = 0\nfor i in range({n}):\n    if i % {b} == {c % b}:\n        total += i * {a}\n    else:\n        total -= {c}\nprint(total)",
        f"xs = [i * i % {a + 5} for i in range({n + 3})]\nprint(sum(x for x in xs if x > {c}) + len(set(xs)))",
        f"def f(k):\n    return k if k < {b} else f(k - {b}) + f(k - 1)\nprint(f({n}))",
        f"s = 'abracadabra' * {c}\nprint(s.count('a') * {a} - s.count('br') + len(s) % {b})",
        f"d = {{}}\nfor i in range({n * 2}):\n    d[i % {b + 1}] = d.get(i % {b + 1}, 0) + i\nprint(max(d.values()) - min(d.values()))",
        f"x, y = {a}, {b}\nfor _ in range({n}):\n    x, y = y, (x + y) % {a + 10}\nprint(x * y)",
    ]
    return rng.choice(templates)


def code_output(n: int, rng: random.Random) -> list[dict[str, Any]]:
    rows, seen = [], set()
    while len(rows) < n:
        prog = _program(rng)
        if prog in seen:
            continue
        seen.add(prog)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(prog, {})  # our own generated arithmetic, not external input
        rows.append({"category": "maths", "source": "code_output",
                     "prompt": f"What does this Python program print?\n\n```python\n{prog}\n```" + SUFFIX,
                     "target_words": None, "gold": buf.getvalue().strip()})
    return rows


def main() -> None:
    rng = random.Random(SEED)
    base = lc.read_jsonl(lc.LAB_DIR / "prompts_train.jsonl")
    hard = math_train_hard(70, rng) + code_output(30, rng)
    held = {r["prompt"] for r in lc.read_jsonl(lc.LAB_DIR / "prompts_hardmath.jsonl")}
    held |= {r["prompt"] for r in lc.read_jsonl(lc.LAB_DIR / "prompts_eval.jsonl")}
    hard = [r for r in hard if r["prompt"] not in held]
    for i, r in enumerate(hard):
        r["id"] = f"hd{i:04d}"
    rows = base + hard
    lc.write_jsonl(lc.LAB_DIR / "prompts_train_hard.jsonl", rows)
    print(f"base={len(base)} hard={len(hard)} share={len(hard) / len(rows):.2f}")


if __name__ == "__main__":
    main()
