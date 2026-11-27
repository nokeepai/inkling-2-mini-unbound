"""Turns the raw public downloads into the four pub suite files; re-runnable, prints one JSON line of counts.

  python3 -I scripts/evalsuite/build_pub.py [--raw ~/.inkling-abl/tinker_lab/pub_raw] \
      [--out ~/.inkling-abl/tinker_lab/pub] [--lab ~/.inkling-abl/tinker_lab] [--suites pub_gsm8k,...]

Raw layout (one directory per dataset, downloaded unauthenticated from the official locations):
  humaneval/HumanEval.jsonl.gz      github.com/openai/human-eval data/HumanEval.jsonl.gz (MIT)
  quixbugs/QuixBugs-master/         github.com/jkoppel/QuixBugs master tarball, extracted (MIT)
  gsm8k/test.jsonl                  github.com/openai/grade-school-math grade_school_math/data/test.jsonl (MIT)
  mmlu/test.parquet                 huggingface.co/datasets/cais/mmlu all/test-00000-of-00001.parquet (MIT)
The raw files are data only: nothing from them is imported or executed here, except that the HumanEval canonical
solutions and the QuixBugs reference programs run once in the code suites' sandbox to verify the harness.
MMLU needs pyarrow to read parquet; without it pub_mmlu is skipped with a note (any Python with pyarrow works).

Release suites (--suites new), each pinned to a revision recorded in every row's `provenance` (see PROVENANCE):
  gsm8k_full/test.jsonl             github.com/openai/grade-school-math @3101c7d (MIT); same bytes as gsm8k/
  mmlu_2k/test.parquet              huggingface.co/datasets/cais/mmlu @c30699e all/test (MIT); same bytes as mmlu/
  mbpp/test.parquet                 huggingface.co/datasets/google-research-datasets/mbpp @4bb6404 sanitized/test
                                    (CC-BY-4.0)
  humanevalpack/python_test.parquet huggingface.co/datasets/bigcode/humanevalpack @9a41762 python/test (MIT)
  math500/test.jsonl                huggingface.co/datasets/HuggingFaceH4/MATH-500 @6e4ed1a test.jsonl
Fetch them anonymously with curl from https://huggingface.co/datasets/<repo>/resolve/<revision>/<path> and
https://raw.githubusercontent.com/openai/grade-school-math/<revision>/grade_school_math/data/test.jsonl.
MBPP, HumanEvalPack and MMLU need pyarrow, e.g. `uv run --no-project --with pyarrow python -I build_pub.py`.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mathcheck as C  # noqa: E402
import metrics as M  # noqa: E402
import pub as P  # noqa: E402

Rec = dict[str, Any]
SEED = 1234                     # one constant for every sample, so the files are reproducible
GSM8K_N = 300
MMLU_N = 400
# The two QuixBugs cases its own pytest suite skips by default (exponential in the reference programs too), plus
# levenshtein's rosettacode case: the exponential reference fix needs ~9.5 s of the sandbox's 10 s, so a correct
# minimal fix would pass or time out depending on machine speed.
QUIX_SLOW = {"knapsack": lambda args: args[0] == 6404180,
             "levenshtein": lambda args: args in (["amanaplanacanalpanama",
                                                   "docnoteidissentafastneverpreventsafatnessidietoncod"],
                                                  ["rosettacode", "raisethysword"])}
QUIX_APPROX = {"sqrt"}          # test_sqrt.py compares with pytest.approx(abs=epsilon)


def write_jsonl(path: Path, rows: list[Rec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[Rec]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def all_pass(jobs: list[tuple[str, bool]]) -> list[bool]:
    """Runs (program, should_pass) through the sandbox; True where the outcome matched."""
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(lambda j: M.run_program(j[0])[0] == j[1], jobs))

# ---------- HumanEval ----------


def humaneval_prompt(code_prompt: str) -> str:
    return ("Complete the following Python function so that it does what its docstring says.\n\n"
            f"```python\n{code_prompt}```\n\n"
            "Reply with the complete function (signature included, plus any imports it needs) "
            "in a single ```python code block.")


def build_humaneval(raw: Path) -> tuple[list[Rec], Rec]:
    src = [json.loads(x) for x in gzip.decompress((raw / "humaneval" / "HumanEval.jsonl.gz").read_bytes())
           .decode().splitlines() if x.strip()]
    rows = [{"id": r["task_id"], "prompt": humaneval_prompt(r["prompt"]), "code_prompt": r["prompt"],
             "entry_point": r["entry_point"], "test": r["test"],
             "reference": r["prompt"] + r["canonical_solution"]} for r in src]
    ok = all_pass([(P.humaneval_program(r, r["reference"]), True) for r in rows])
    return rows, {"canonical_pass": sum(ok), "canonical_fail_ids": [r["id"] for r, k in zip(rows, ok) if not k]}

# ---------- QuixBugs ----------


def quixbugs_prompt(name: str, buggy: str) -> str:
    return (f"The following Python program has exactly one bug: one line is wrong. Find it and fix it.\n\n"
            f"```python\n{buggy}```\n\n"
            f"Reply with the COMPLETE corrected program in a single ```python code block. Keep the function "
            f"name `{name}` and its signature unchanged.")


def quix_cases(name: str, path: Path) -> list[Any]:
    cases = read_jsonl(path)
    slow = QUIX_SLOW.get(name)
    return [c for c in cases if not (slow and slow(c[0] if isinstance(c[0], list) else [c[0]]))]


def build_quixbugs(raw: Path) -> tuple[list[Rec], Rec]:
    root = raw / "quixbugs" / "QuixBugs-master"
    rows = []
    for f in sorted((root / "json_testcases").glob("*.json")):
        name = f.stem
        buggy = (root / "python_programs" / f"{name}.py").read_text()
        rows.append({"id": f"quixbugs/{name}", "entry_point": name, "prompt": quixbugs_prompt(name, buggy),
                     "buggy": buggy, "reference": (root / "correct_python_programs" / f"{name}.py").read_text(),
                     "approx": name in QUIX_APPROX, "cases": quix_cases(name, f)})
    jobs = [(P.quixbugs_program(r, r["reference"]), True) for r in rows]
    jobs += [(P.quixbugs_program(r, r["buggy"]), False) for r in rows]
    ok = all_pass(jobs)
    good_fix, bad_fails = ok[:len(rows)], ok[len(rows):]
    keep = [r for r, a, b in zip(rows, good_fix, bad_fails) if a and b]
    return keep, {"programs_with_json_cases": len(rows), "kept": len(keep),
                  "reference_failed": [r["entry_point"] for r, a in zip(rows, good_fix) if not a],
                  "buggy_passed": [r["entry_point"] for r, b in zip(rows, bad_fails) if not b],
                  "no_json_cases_skipped": len(list((root / "python_programs").glob("*_test.py")))}

# ---------- GSM8K ----------


def lab_texts(lab: Path) -> set[str]:
    """Every prompt in the project's own files: GSM8K items already used there (hardmath) are not 'unseen'."""
    out = set()
    for f in lab.glob("*.jsonl"):
        for r in read_jsonl(f):
            if isinstance(r.get("prompt"), str):
                out.add(squash(r["prompt"]))
    return out


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def build_gsm8k(raw: Path, lab: Path) -> tuple[list[Rec], Rec]:
    src = read_jsonl(raw / "gsm8k" / "test.jsonl")
    seen = lab_texts(lab)
    blob = "\n".join(seen)
    pool = [(i, r) for i, r in enumerate(src) if squash(r["question"]) not in blob]
    pick = sorted(random.Random(SEED).sample(pool, GSM8K_N), key=lambda x: x[0])
    rows = [{"id": f"gsm8k_test/{i}", "category": "maths", "source": "gsm8k_test", "target_words": None,
             "prompt": f"{r['question']}\n\nGive the final answer in \\boxed{{}}.",
             "gold": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in pick]
    return rows, {"test_items": len(src), "excluded_seen_in_lab_files": len(src) - len(pool), "seed": SEED}

# ---------- MMLU ----------


def mmlu_prompt(r: Rec) -> str:
    opts = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", r["choices"]))
    subject = r["subject"].replace("_", " ")
    return (f"The following is a multiple choice question about {subject}.\n\n{r['question']}\n\n{opts}\n\n"
            "Think it through, then end your reply with a final line of exactly the form \"Answer: X\", "
            "where X is A, B, C or D.")


def stratified(src: list[Rec], n: int, rng: random.Random) -> list[tuple[int, Rec]]:
    """Equal share per subject, the remainder from seeded-random subjects, then a seeded shuffle so a
    --limit smoke run sees many subjects."""
    by: dict[str, list[tuple[int, Rec]]] = defaultdict(list)
    for i, r in enumerate(src):
        by[r["subject"]].append((i, r))
    subjects = sorted(by)
    per, extra = divmod(n, len(subjects))
    pick = []
    for s in subjects:
        pick += rng.sample(by[s], per)
    chosen = {i for i, _ in pick}
    for s in rng.sample(subjects, extra):
        pick.append(rng.choice([x for x in by[s] if x[0] not in chosen]))
    rng.shuffle(pick)
    return pick


def build_mmlu(raw: Path) -> tuple[list[Rec], Rec]:
    import pyarrow.parquet as pq  # only this suite needs it; ImportError is reported by the caller
    src = [r for r in pq.read_table(raw / "mmlu" / "test.parquet").to_pylist() if len(r["choices"]) == 4]
    pick = stratified(src, MMLU_N, random.Random(SEED))
    rows = [{"id": f"mmlu/{r['subject']}/{i}", "subject": r["subject"], "prompt": mmlu_prompt(r),
             "gold": "ABCD"[int(r["answer"])]} for i, r in pick]
    return rows, {"test_items": len(src), "subjects": len({r["subject"] for r in src}), "seed": SEED}


# ---------- release suites: full or large public sets, pinned revisions, provenance on every row ----------

MMLU_2K_N = 2000
# What every row of a release suite records. `file_sha256` is filled in from the raw file at build time.
PROVENANCE: dict[str, Rec] = {
    "pub_gsm8k_full": {"dataset": "GSM8K", "repo": "github.com/openai/grade-school-math",
                       "path": "grade_school_math/data/test.jsonl", "config": None, "split": "test",
                       "revision": "3101c7d5072418e28b9008a6636bde82a006892c", "licence": "MIT",
                       "sampling": "all", "seed": None},
    "pub_mmlu_2k": {"dataset": "MMLU", "repo": "huggingface.co/datasets/cais/mmlu",
                    "path": "all/test-00000-of-00001.parquet", "config": "all", "split": "test",
                    "revision": "c30699e8356da336a370243923dbaf21066bb9fe", "licence": "MIT",
                    "sampling": "2,000 stratified proportionally across the 57 subjects (largest remainder)",
                    "seed": SEED},
    "pub_mbpp": {"dataset": "MBPP (sanitized)", "repo": "huggingface.co/datasets/google-research-datasets/mbpp",
                 "path": "sanitized/test-00000-of-00001.parquet", "config": "sanitized", "split": "test",
                 "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "licence": "CC-BY-4.0",
                 "sampling": "all", "seed": None},
    "pub_humanevalfix": {"dataset": "HumanEvalPack HumanEvalFix (Python)",
                         "repo": "huggingface.co/datasets/bigcode/humanevalpack",
                         "path": "python/test-00000-of-00001.parquet", "config": "python", "split": "test",
                         "revision": "9a41762f73a8cb23bb5811b73d5aab164efcf378", "licence": "MIT",
                         "sampling": "all", "seed": None,
                         "prompt_form": "fix-tests, bigcode-evaluation-harness humanevalfixtests-python 'instruct'"},
    "pub_math500": {"dataset": "MATH-500 (PRM800K test subset of Hendrycks MATH)",
                    "repo": "huggingface.co/datasets/HuggingFaceH4/MATH-500", "path": "test.jsonl",
                    "config": None, "split": "test", "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                    "licence": "MIT (Hendrycks MATH and openai/prm800k are MIT; the HF card states none)",
                    "sampling": "all", "seed": None},
}
RAW_FILES = {"pub_gsm8k_full": "gsm8k_full/test.jsonl", "pub_mmlu_2k": "mmlu_2k/test.parquet",
             "pub_mbpp": "mbpp/test.parquet", "pub_humanevalfix": "humanevalpack/python_test.parquet",
             "pub_math500": "math500/test.jsonl"}
LAB_MIN_CHARS = 60  # shorter texts ("Which of the following is true?") would match the lab files by accident


def provenance(suite: str, raw: Path) -> Rec:
    digest = hashlib.sha256((raw / RAW_FILES[suite]).read_bytes()).hexdigest()
    return {**PROVENANCE[suite], "file_sha256": digest}


def read_parquet(path: Path) -> list[Rec]:
    import pyarrow.parquet as pq  # ImportError is reported by main() as a skipped suite
    return pq.read_table(path).to_pylist()


def lab_blobs(lab: Path) -> dict[str, str]:
    """Squashed prompt text of each project prompt file, so every release row can say where it was seen."""
    out = {}
    for f in sorted(lab.glob("*.jsonl")):
        texts = [squash(r["prompt"]) for r in read_jsonl(f) if isinstance(r.get("prompt"), str)]
        out[f.name] = "\n".join(texts)
    return out


def seen_in(text: str, blobs: dict[str, str]) -> list[str]:
    t = squash(text)
    return [] if len(t) < LAB_MIN_CHARS else [name for name, blob in blobs.items() if t in blob]


def lab_summary(rows: list[Rec]) -> Rec:
    files: dict[str, int] = defaultdict(int)
    for r in rows:
        for f in r["seen_in_lab_files"]:
            files[f] += 1
    return {"rows_seen_in_lab_files": sum(bool(r["seen_in_lab_files"]) for r in rows), "by_file": dict(files)}


def build_gsm8k_full(raw: Path, blobs: dict[str, str]) -> tuple[list[Rec], Rec]:
    """All 1,319 test items, same prompt, gold and id as pub_gsm8k so the two share items and results pair."""
    src = read_jsonl(raw / RAW_FILES["pub_gsm8k_full"])
    prov = provenance("pub_gsm8k_full", raw)
    rows = [{"id": f"gsm8k_test/{i}", "category": "maths", "source": "gsm8k_test", "target_words": None,
             "prompt": f"{r['question']}\n\nGive the final answer in \\boxed{{}}.",
             "gold": r["answer"].split("####")[-1].strip().replace(",", ""),
             "seen_in_lab_files": seen_in(r["question"], blobs), "provenance": prov} for i, r in enumerate(src)]
    return rows, {"test_items": len(src), **lab_summary(rows)}


def proportional(src: list[Rec], n: int, rng: random.Random) -> list[tuple[int, Rec]]:
    """n items split across subjects in proportion to their test-set size (largest remainder, ties by name),
    sampled per subject in name order with one seeded generator, then shuffled for --limit smoke runs."""
    by: dict[str, list[tuple[int, Rec]]] = defaultdict(list)
    for i, r in enumerate(src):
        by[r["subject"]].append((i, r))
    exact = {s: n * len(v) / len(src) for s, v in by.items()}
    quota = {s: int(q) for s, q in exact.items()}
    by_remainder = sorted(by, key=lambda s: (-(exact[s] - quota[s]), s))
    for s in by_remainder[: n - sum(quota.values())]:
        quota[s] += 1
    pick = [x for s in sorted(by) for x in rng.sample(by[s], quota[s])]
    rng.shuffle(pick)
    return pick


def build_mmlu_2k(raw: Path, blobs: dict[str, str]) -> tuple[list[Rec], Rec]:
    src = [r for r in read_parquet(raw / RAW_FILES["pub_mmlu_2k"]) if len(r["choices"]) == 4]
    prov = provenance("pub_mmlu_2k", raw)
    pick = proportional(src, MMLU_2K_N, random.Random(SEED))
    rows = [{"id": f"mmlu/{r['subject']}/{i}", "subject": r["subject"], "prompt": mmlu_prompt(r),
             "gold": "ABCD"[int(r["answer"])], "seen_in_lab_files": seen_in(r["question"], blobs),
             "provenance": prov} for i, r in pick]
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["subject"]] += 1
    return rows, {"test_items": len(src), "subjects": len(counts), "seed": SEED,
                  "per_subject": dict(sorted(counts.items())), **lab_summary(rows)}


def mbpp_prompt(text: str, tests: list[str]) -> str:
    """The MBPP paper's task wording (Austin et al. 2021, all three asserts shown), plus the reply-format line
    every pub code suite uses so the last ```python block is the answer."""
    shown = "\n".join(tests)
    return (f"You are an expert Python programmer, and here is your task: {text} "
            f"Your code should pass these tests:\n\n{shown}\n\n"
            "Reply with the complete Python code (any imports included) in a single ```python code block.")


def build_mbpp(raw: Path, blobs: dict[str, str]) -> tuple[list[Rec], Rec]:
    src = read_parquet(raw / RAW_FILES["pub_mbpp"])
    prov = provenance("pub_mbpp", raw)
    rows = [{"id": f"mbpp/{r['task_id']}", "prompt": mbpp_prompt(r["prompt"], list(r["test_list"])),
             "test_imports": list(r["test_imports"]), "tests": list(r["test_list"]), "reference": r["code"],
             "seen_in_lab_files": seen_in(r["prompt"], blobs), "provenance": prov} for r in src]
    ok = all_pass([(P.mbpp_program(r, r["reference"]), True) for r in rows])
    return rows, {"test_items": len(src), "reference_pass": sum(ok),
                  "reference_fail_ids": [r["id"] for r, k in zip(rows, ok) if not k], **lab_summary(rows)}


def humanevalfix_prompt(r: Rec) -> str:
    """bigcode-evaluation-harness HumanEvalFixTests, prompt='instruct' (humanevalpack.py at 8fc5bae): context =
    declaration + buggy body + "\\n" + the unit tests, then "Fix bugs in <entry_point>.". The harness appends the
    declaration for a completion model to continue; a chat model is asked for the whole function instead."""
    context = r["declaration"] + r["buggy_solution"] + "\n" + r["test"]
    instruct = (context + "\n" + f"Fix bugs in {r['entry_point']}.").strip()
    return (f"{instruct}\n\nReply with the complete fixed function (signature included, plus any imports it "
            "needs) in a single ```python code block.")


def build_humanevalfix(raw: Path, blobs: dict[str, str]) -> tuple[list[Rec], Rec]:
    src = read_parquet(raw / RAW_FILES["pub_humanevalfix"])
    prov = provenance("pub_humanevalfix", raw)
    rows = [{"id": f"humanevalfix/{r['task_id']}", "prompt": humanevalfix_prompt(r), "code_prompt": r["prompt"],
             "entry_point": r["entry_point"], "test": r["test"], "bug_type": r["bug_type"],
             "reference": r["declaration"] + r["canonical_solution"],
             "buggy": r["declaration"] + r["buggy_solution"],
             "seen_in_lab_files": seen_in(r["buggy_solution"], blobs), "provenance": prov} for r in src]
    jobs = [(P.humanevalfix_program(r, r["reference"]), True) for r in rows]
    jobs += [(P.humanevalfix_program(r, r["buggy"]), False) for r in rows]
    ok = all_pass(jobs)
    good, bad = ok[:len(rows)], ok[len(rows):]
    return rows, {"test_items": len(src), "fixed_pass": sum(good), "buggy_fail": sum(bad),
                  "fixed_failed_ids": [r["id"] for r, k in zip(rows, good) if not k],
                  "buggy_passed_ids": [r["id"] for r, k in zip(rows, bad) if not k], **lab_summary(rows)}


def build_math500(raw: Path, blobs: dict[str, str]) -> tuple[list[Rec], Rec]:
    src = read_jsonl(raw / RAW_FILES["pub_math500"])
    prov = provenance("pub_math500", raw)
    rows = [{"id": f"math500/{r['unique_id']}", "category": "maths", "subject": r["subject"],
             "level": int(r["level"]), "prompt": f"{r['problem']}\n\nGive the final answer in \\boxed{{}}.",
             "gold": r["answer"], "seen_in_lab_files": seen_in(r["problem"], blobs), "provenance": prov}
            for r in src]
    # The harness check: every gold answer must be equivalent to the box in its own reference solution.
    ok = sum(C.math_correct(r["solution"], r["answer"]) for r in src)
    return rows, {"test_items": len(src), "gold_matches_reference_box": ok, **lab_summary(rows)}


def new_builders(raw: Path, lab: Path) -> dict[str, Callable[[], tuple[list[Rec], Rec]]]:
    blobs: dict[str, str] = {}

    def lazy() -> dict[str, str]:  # the lab files are read once, and only when a release suite is built
        if not blobs:
            blobs.update(lab_blobs(lab))
        return blobs
    return {"pub_gsm8k_full": lambda: build_gsm8k_full(raw, lazy()), "pub_mmlu_2k": lambda: build_mmlu_2k(raw, lazy()),
            "pub_mbpp": lambda: build_mbpp(raw, lazy()), "pub_humanevalfix": lambda: build_humanevalfix(raw, lazy()),
            "pub_math500": lambda: build_math500(raw, lazy())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="~/.inkling-abl/tinker_lab/pub_raw")
    ap.add_argument("--out", default="~/.inkling-abl/tinker_lab/pub")
    ap.add_argument("--lab", default="~/.inkling-abl/tinker_lab", help="project prompt files (GSM8K exclusion)")
    ap.add_argument("--suites", default=",".join(P.SUITE_NAMES),
                    help="suite names, or 'new' (the five release suites) or 'all' (all nine)")
    a = ap.parse_args()
    raw, out, lab = (Path(x).expanduser() for x in (a.raw, a.out, a.lab))
    builders: dict[str, Callable[[], tuple[list[Rec], Rec]]] = {
        "pub_humaneval": lambda: build_humaneval(raw), "pub_quixbugs": lambda: build_quixbugs(raw),
        "pub_gsm8k": lambda: build_gsm8k(raw, lab), "pub_mmlu": lambda: build_mmlu(raw),
        **new_builders(raw, lab)}
    names = {"new": ",".join(P.NEW_SUITE_NAMES), "all": ",".join(P.SUITE_NAMES + P.NEW_SUITE_NAMES)}
    report: Rec = {}
    for suite in ",".join(names.get(x, x) for x in a.suites.split(",")).split(","):
        try:
            rows, info = builders[suite]()
        except (ImportError, FileNotFoundError) as e:
            report[suite] = {"skipped": f"{type(e).__name__}: {e}"}
            continue
        write_jsonl(out / f"{suite}.jsonl", rows)
        report[suite] = {"items": len(rows), **info}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
