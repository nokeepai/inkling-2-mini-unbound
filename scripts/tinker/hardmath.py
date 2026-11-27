"""Hard maths eval set: 50 GSM8K test + 50 MATH test level 4-5 (integer answers), fixed seed.
Test splits only, so it is disjoint from the training pool (GSM8K/MATH train splits).

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/hardmath.py
  $PY scripts/tinker/eval.py --tag base_hardmath --prompts prompts_hardmath.jsonl
"""
from __future__ import annotations

import random
import re
from typing import Any

import lab_common as lc
from prompts import BLOCKLIST, _last_boxed

SEED = 4242
SUFFIX = "\n\nGive the final answer in \\boxed{}."


def gsm8k_test(n: int, rng: random.Random) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i in rng.sample(range(len(ds)), n * 2):
        if BLOCKLIST.search(ds[i]["question"]):
            continue
        rows.append({"category": "maths", "source": "gsm8k_test", "prompt": ds[i]["question"] + SUFFIX,
                     "target_words": None,
                     "gold": ds[i]["answer"].split("####")[-1].strip().replace(",", "")})
    return rows[:n]


def math_hard(n: int, rng: random.Random) -> list[dict[str, Any]]:
    from datasets import load_dataset

    pool = []
    for s in ["algebra", "prealgebra", "number_theory", "counting_and_probability",
              "intermediate_algebra", "geometry", "precalculus"]:
        for r in load_dataset("EleutherAI/hendrycks_math", s, split="test"):
            gold = _last_boxed(r["solution"])
            if r["level"] in ("Level 4", "Level 5") and gold and re.fullmatch(r"-?\d+", gold) \
                    and not BLOCKLIST.search(r["problem"]):
                pool.append({"category": "maths", "source": f"math_test_{s}_{r['level'][-1]}",
                             "prompt": r["problem"] + SUFFIX, "target_words": None, "gold": gold})
    return rng.sample(pool, n)


def main() -> None:
    rng = random.Random(SEED)
    rows = gsm8k_test(50, rng) + math_hard(50, rng)
    train = {r["prompt"] for r in lc.read_jsonl(lc.LAB_DIR / "prompts_train.jsonl")}
    assert not any(r["prompt"] in train for r in rows), "overlap with training pool"
    for i, r in enumerate(rows):
        r["id"] = f"hm{i:03d}"
    lc.write_jsonl(lc.LAB_DIR / "prompts_hardmath.jsonl", rows)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
