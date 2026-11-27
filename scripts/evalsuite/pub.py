"""Public benchmark suites the training recipe was never tuned against: HumanEval, QuixBugs, GSM8K, MMLU, and the
release suites (full GSM8K, MMLU 2k, MBPP sanitized, HumanEvalFix Python, MATH-500).

Every suite is scored deterministically (unit tests, gold number, or option letter): no judge, no cost, no grader
bias. Items come from <data-dir>/pub/<suite>.jsonl, written by build_pub.py from the raw public downloads.
Generation goes through the same client call, caps, effort and per-item seeds as the lab's single-turn suites,
and model code runs in the same sandbox as the `code` suite (metrics.run_program).
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable

import mathcheck as C
import metrics as M
import suites as S

Rec = dict[str, Any]
PUB_DIR = "pub"
SUITE_NAMES = ("pub_humaneval", "pub_quixbugs", "pub_gsm8k", "pub_mmlu")
# Release suites: full or large public sets for confidence intervals. A separate tuple (and run_suite.py alias
# pub_new) so the `pub` alias, its data and earlier results stay exactly as they were.
NEW_SUITE_NAMES = ("pub_gsm8k_full", "pub_mmlu_2k", "pub_mbpp", "pub_humanevalfix", "pub_math500")
# The sandbox timeout is wall time, so dozens of programs at once on a few cores can time out correct code
# (seen with 64 workers on a laptop). Bounding concurrent runs keeps the 10 s limit about the program alone.
SANDBOX_SLOTS = threading.BoundedSemaphore(max(1, (os.cpu_count() or 2) // 2))


def pub_path(data_dir: Path, suite: str) -> Path:
    return data_dir / PUB_DIR / f"{suite}.jsonl"


def load_pub(data_dir: Path, suite: str) -> list[Rec]:
    path = pub_path(data_dir, suite)
    if not path.exists():
        raise SystemExit(f"{suite}: missing {path}. Build it with scripts/evalsuite/build_pub.py and copy "
                         f"~/.inkling-abl/tinker_lab/pub/ to <data-dir>/{PUB_DIR}/ (README.md, public suites).")
    return S.read_jsonl(path)

# ---------- programs run in the sandbox ----------


def humaneval_program(row: Rec, code: str) -> str:
    """The dataset prompt first, so helper functions and imports it defines exist even when the model
    returns only the target function; the model's definition then replaces the docstring-only stub."""
    return f"{row['code_prompt']}\n\n{code}\n\n{row['test']}\n\ncheck({row['entry_point']})\n"


# Mirrors QuixBugs' tester.py / python_testcases: a non-list input is one argument, generators are list()-ed,
# tuples compare as lists (test_hanoi turns expected rows into tuples; JSON has none), sqrt is approximate
# within its epsilon argument, and an exception fails the case.
QUIX_HARNESS = '''

import copy as _qb_copy, json as _qb_json, types as _qb_types


def _qb_norm(x):
    if isinstance(x, (_qb_types.GeneratorType, list, tuple)):
        return [_qb_norm(v) for v in x]
    return x


def _qb_same(got, want, args, approx):
    if approx:
        return isinstance(got, (int, float)) and abs(got - want) <= args[-1]
    return got == want


_qb_fails = []
for _qb_i, (_qb_in, _qb_out) in enumerate(_qb_json.loads({cases})):
    _qb_args = _qb_in if isinstance(_qb_in, list) else [_qb_in]
    try:
        _qb_got = _qb_norm({entry}(*_qb_copy.deepcopy(_qb_args)))
    except Exception as _qb_e:
        _qb_got = repr(_qb_e)
    if not _qb_same(_qb_got, _qb_norm(_qb_out), _qb_args, {approx}):
        _qb_fails.append(_qb_i)
if _qb_fails:
    raise SystemExit("failed cases: %s" % _qb_fails)
'''


def quixbugs_program(row: Rec, code: str) -> str:
    return code + QUIX_HARNESS.format(cases=repr(json.dumps(row["cases"])), entry=row["entry_point"],
                                      approx=bool(row.get("approx")))


def mbpp_program(row: Rec, code: str) -> str:
    """MBPP's test_imports (e.g. `import math` for math.isclose asserts), the model's code, then every assert."""
    return "\n".join(row["test_imports"]) + f"\n{code}\n\n" + "\n".join(row["tests"]) + "\n"


# bigcode-evaluation-harness prepends these to every Python HumanEvalPack program (humanevalpack.py,
# IMPORT_HELPER["python"]). numpy goes in a try so a sandbox without it does not fail every item.
HEF_IMPORTS = "\n".join(f"import {m}" for m in (
    "math", "re", "sys", "copy", "datetime", "itertools", "collections", "heapq", "statistics", "functools",
    "hashlib", "string")) + "\nfrom typing import *\nfrom collections import *\n" \
    "try:\n    import numpy\n    import numpy as np\nexcept ImportError:\n    pass\n"


def humanevalfix_program(row: Rec, code: str) -> str:
    """The harness's imports, the dataset prompt (helpers the declaration defines, e.g. is_palindrome, exist
    even when the model returns only the target function), the model's function, then the hidden tests, which
    end in check(entry_point) themselves."""
    return f"{HEF_IMPORTS}\n{row['code_prompt']}\n\n{code}\n\n{row['test']}\n"

# ---------- code suites ----------


def run_sandboxed(program: str) -> tuple[bool, str]:
    with SANDBOX_SLOTS:
        return M.run_program(program)


def gen_pub_code(ctx: S.Ctx, suite: str, build: Callable[[Rec, str], str]) -> list[Rec]:
    rows = S.limited(ctx, load_pub(ctx.data_dir, suite))

    def one(row: Rec) -> Rec:
        g = ctx.client.generate([{"role": "user", "content": row["prompt"]}], S.seed_for(ctx, row["id"]))
        code = M.last_code_block(g.answer)
        ok, err = run_sandboxed(build(row, code)) if code is not None else (False, "no code block")
        return {"id": row["id"], "passed": ok, "err": err, **S.gen_fields(g),
                "thinking": g.thinking, "answer": g.answer}
    return S.pmap(one, rows, ctx.workers)

# ---------- GSM8K: the hardmath path (gen_single + maths_quality on \\boxed{}) ----------


def gen_pub_gsm8k(ctx: S.Ctx, suite: str = "pub_gsm8k") -> list[Rec]:
    load_pub(ctx.data_dir, suite)  # clear message when missing; gen_single reads the file itself
    return S.gen_single(ctx, f"{PUB_DIR}/{suite}.jsonl")

# ---------- MATH-500: MATH's own boxed extraction and is_equiv (mathcheck.py) ----------


def gen_pub_math500(ctx: S.Ctx) -> list[Rec]:
    rows = S.limited(ctx, load_pub(ctx.data_dir, "pub_math500"))

    def one(row: Rec) -> Rec:
        g = ctx.client.generate([{"role": "user", "content": row["prompt"]}], S.seed_for(ctx, row["id"]))
        pred = C.last_boxed(g.answer)
        correct = C.is_equiv(pred, row["gold"])
        gold_num, pred_num = C.gold_number(row["gold"]), C.number_value(pred)
        number = correct or (gold_num is not None and pred_num is not None and C.same_number(pred_num, gold_num))
        return {"id": row["id"], "subject": row["subject"], "level": row["level"], "gold": row["gold"],
                "correct": correct, "correct_number": number, "no_box": pred is None, **S.gen_fields(g),
                "thinking": g.thinking, "answer": g.answer}
    return S.pmap(one, rows, ctx.workers)


def sum_pub_math500(recs: list[Rec]) -> Rec:
    ok = S.ok_rows(recs)
    if not ok:
        return {"n": 0, "n_failed": len(recs)}
    by = {k: sorted({r[k] for r in ok}) for k in ("subject", "level")}
    return {"n": len(ok), "n_failed": len(recs) - len(ok), "accuracy": M.rate(ok, "correct"),
            "accuracy_number": M.rate(ok, "correct_number"), "no_box": M.rate(ok, "no_box"),
            "cap_hit": M.rate(ok, "think_cap_hit"), "runaway": M.rate(ok, "answer_runaway"),
            "loop_any": M.rate(ok, "loop_any"), "median_think_tokens": M.median([r["think_tokens"] for r in ok]),
            "median_answer_tokens": M.median([r["answer_tokens"] for r in ok]),
            "by_subject": {s: M.rate([r for r in ok if r["subject"] == s], "correct") for s in by["subject"]},
            "by_level": {str(lv): M.rate([r for r in ok if r["level"] == lv], "correct") for lv in by["level"]}}

# ---------- MMLU ----------


ANSWER_WORD_RE = re.compile(r"answer", re.I)
CHOICE_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def extract_choice(answer: str) -> str | None:
    """Last standalone A-D after the last 'answer' on the last line that says 'answer'; None if absent,
    so a reply without an answer line earns nothing rather than a guess from stray letters."""
    lines = [ln for ln in answer.splitlines() if ANSWER_WORD_RE.search(ln)]
    if not lines:
        return None
    line = lines[-1]
    tail = line[[m.end() for m in ANSWER_WORD_RE.finditer(line)][-1]:]
    letters = CHOICE_RE.findall(tail)
    return letters[-1] if letters else None


def gen_pub_mmlu(ctx: S.Ctx, suite: str = "pub_mmlu") -> list[Rec]:
    rows = S.limited(ctx, load_pub(ctx.data_dir, suite))

    def one(row: Rec) -> Rec:
        g = ctx.client.generate([{"role": "user", "content": row["prompt"]}], S.seed_for(ctx, row["id"]))
        pred = extract_choice(g.answer)
        return {"id": row["id"], "subject": row["subject"], "gold": row["gold"], "pred": pred,
                "correct": pred == row["gold"], "no_answer": pred is None, **S.gen_fields(g),
                "thinking": g.thinking, "answer": g.answer}
    return S.pmap(one, rows, ctx.workers)


def sum_pub_mmlu(recs: list[Rec]) -> Rec:
    ok = S.ok_rows(recs)
    if not ok:
        return {"n": 0, "n_failed": len(recs)}
    subjects = sorted({r["subject"] for r in ok})
    return {"n": len(ok), "n_failed": len(recs) - len(ok), "accuracy": M.rate(ok, "correct"),
            "no_answer": M.rate(ok, "no_answer"), "cap_hit": M.rate(ok, "think_cap_hit"),
            "runaway": M.rate(ok, "answer_runaway"), "loop_any": M.rate(ok, "loop_any"),
            "median_think_tokens": M.median([r["think_tokens"] for r in ok]),
            "median_answer_tokens": M.median([r["answer_tokens"] for r in ok]),
            "by_subject": {s: M.rate([r for r in ok if r["subject"] == s], "correct") for s in subjects},
            "by_subject_n": {s: sum(r["subject"] == s for r in ok) for s in subjects}}


SUITES: dict[str, tuple[Callable[[S.Ctx], list[Rec]], Callable[[S.Ctx, list[Rec]], dict[str, int]],
                        Callable[[list[Rec]], Rec]]] = {
    "pub_humaneval": (lambda c: gen_pub_code(c, "pub_humaneval", humaneval_program), S._no_judge, S.sum_code),
    "pub_quixbugs": (lambda c: gen_pub_code(c, "pub_quixbugs", quixbugs_program), S._no_judge, S.sum_code),
    "pub_gsm8k": (gen_pub_gsm8k, S._no_judge, S.sum_single),
    "pub_mmlu": (gen_pub_mmlu, S._no_judge, sum_pub_mmlu),
    "pub_gsm8k_full": (lambda c: gen_pub_gsm8k(c, "pub_gsm8k_full"), S._no_judge, S.sum_single),
    "pub_mmlu_2k": (lambda c: gen_pub_mmlu(c, "pub_mmlu_2k"), S._no_judge, sum_pub_mmlu),
    "pub_mbpp": (lambda c: gen_pub_code(c, "pub_mbpp", mbpp_program), S._no_judge, S.sum_code),
    "pub_humanevalfix": (lambda c: gen_pub_code(c, "pub_humanevalfix", humanevalfix_program), S._no_judge,
                         S.sum_code),
    "pub_math500": (gen_pub_math500, S._no_judge, sum_pub_math500),
}


def checks() -> None:
    assert extract_choice("so...\nAnswer: B") == "B"
    assert extract_choice("**Final answer:** (C)") == "C"
    assert extract_choice("Answer: \\boxed{D}\n\nThat is all.") == "D"
    assert extract_choice("The answer is A.") == "A"
    assert extract_choice("I pick B") is None
    assert extract_choice("Answer: none of these") is None
    row = {"test_imports": ["import math"], "tests": ["assert math.isclose(f(2), 4.0)"]}
    assert M.run_program(mbpp_program(row, "def f(x):\n    return x * 2"))[0]
    assert not M.run_program(mbpp_program(row, "def f(x):\n    return x + 3"))[0]
    fix = {"code_prompt": "def g(xs: List[int]) -> int:\n    \"\"\"Sum.\"\"\"\n",
           "test": "def check(g):\n    assert g([1, 2]) == 3\n\ncheck(g)\n"}
    assert M.run_program(humanevalfix_program(fix, "def g(xs):\n    return sum(xs)"))[0]
    assert not M.run_program(humanevalfix_program(fix, "def g(xs):\n    return max(xs)"))[0]
