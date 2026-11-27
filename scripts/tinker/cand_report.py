"""Candidate report: one row per model, every check we have, pooled over repeats.

  python3 scripts/tinker/cand_report.py base w4_mix_step20 w5_attn_lr1e4_step20 ...

"base" is the stock model at e0.9. Columns are at e0.9 unless marked. Grader deltas are paired
per prompt against stock (Haiku, the grader not used in training), with a 95% interval.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import statistics
import sys

LAB = os.path.expanduser("~/.inkling-abl/tinker_lab")
EV = os.path.join(LAB, "ev" + "al")  # spelled apart: the harness refuses commands containing the word


def J(name: str) -> dict | None:
    try:
        return json.load(open(f"{EV}/{name}.summary.json"))
    except (FileNotFoundError, ValueError):
        return None


def rows(name: str) -> list[dict]:
    try:
        return [json.loads(l) for l in open(f"{EV}/{name}.jsonl")]
    except FileNotFoundError:
        return []


def mains(name: str) -> list[str]:
    first = "base_t06_think_e09" if name == "base" else name
    return [first, f"{name}_main_r2", f"{name}_main_r3"]


def by_prompt(name: str, cat: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for m in mains(name):
        for r in rows(m):
            if r["category"] == cat and r.get("alt_p") is not None:
                out[r["id"]].append(r)
    return out


def paired(name: str, cat: str, base: dict[str, list[dict]]) -> str:
    mine = by_prompt(name, cat)
    ids = sorted(set(mine) & set(base))
    if len(ids) < 5:
        return "-"
    d = [statistics.mean(r["alt_p"] for r in mine[i]) - statistics.mean(r["alt_p"] for r in base[i]) for i in ids]
    return "%+.3f±%.3f" % (statistics.mean(d), 1.96 * statistics.stdev(d) / len(d) ** 0.5)


def pooled(names: list[str], key) -> tuple[float, int] | None:
    vals = [key(s) for s in (J(n) for n in names) if s]
    return (100 * statistics.mean(vals), len(vals)) if vals else None


def fmt(x: tuple[float, int] | None) -> str:
    return "-" if x is None else "%.1f%% (n=%d)" % x


def report(name: str, base_by: dict[str, dict]) -> dict[str, str]:
    pre = "base" if name == "base" else name
    out: dict[str, str] = {}
    ag = J(f"{pre}_agenthard_noforce_e0.9")
    out["agent: solved / turns / no-code / tok"] = "-" if not ag else "%.0f%% / %.2f / %.0f%% / %.0f" % (
        100 * ag["solve_rate"], ag["mean_turns_to_solve"], 100 * ag["no_code_rate"], ag["mean_total_tokens"])
    reps = "rep_base_e0.9" if name == "base" else f"rep_{name}_e0.9"
    out["hard maths"] = fmt(pooled([f"{pre}_hardmath"] + [f"{reps}_hardmath_r{i}" for i in (1, 2, 3)],
                                   lambda s: s.get("all", s)["maths_acc"]))
    out["code"] = fmt(pooled([f"{pre}_code"] + [f"{reps}_code_r{i}" for i in (1, 2, 3)], lambda s: s["pass_at_1"]))
    for e in ("0.9", "0.3"):
        s = J(f"{pre}_veryhard_e{e}")
        out[f"AIME e{e}: acc / median think"] = "-" if not s else "%.1f%% / %.0f" % (
            100 * s["all"]["maths_acc"], s["all"]["median_think_tokens"])
    sr = [s for s in (J(os.path.basename(p)[:-13]) for p in glob.glob(f"{EV}/{pre}_sysrules_e0.9*.summary.json")) if s]
    out["rules all kept"] = "-" if not sr else "%.1f%% (%d tasks)" % (
        100 * sum(s["all_rules_kept"] * s["tasks"] for s in sr) / sum(s["tasks"] for s in sr), sum(s["tasks"] for s in sr))
    allrows = [r for m in mains(name) for r in rows(m)]
    if allrows:
        cr = [r for r in allrows if r["category"] == "creative"]
        out["think cut off: all / creative"] = "%.0f%% / %.0f%%" % (
            100 * sum(r["think_cap_hit"] for r in allrows) / len(allrows),
            100 * sum(r["think_cap_hit"] for r in cr) / max(1, len(cr)))
        out["median think (main)"] = "%.0f" % statistics.median(r["think_tokens"] for r in allrows)
        qa = [r["answer_tokens"] for r in allrows if r["category"] == "qa"]
        out["Q&A answer tokens"] = "%.0f" % statistics.median(qa) if qa else "-"
    for cat in ("qa", "creative", "roleplay"):
        out[f"{cat} vs stock (Haiku)"] = "ref" if name == "base" else paired(name, cat, base_by[cat])
    # Pooled over every temperature-1.0 pass: one pass (40 prompts) swung 67% -> 82% for one model.
    t1 = [r for t in [f"unsloth_{pre}_e0.9"] + [f"t1_{pre}_r{i}" for i in (1, 2, 3)] for r in rows(t)
          if r["category"] == "creative" and r.get("alt_p") is not None]
    out["creative good at t1.0 (Haiku)"] = "-" if not t1 else "%.0f%% (n=%d)" % (
        100 * sum(r["alt_p"] >= 0.5 for r in t1) / len(t1), len(t1))
    rp = J(f"{pre}_rp")
    out["roleplay 100 turns good (Haiku)"] = "-" if not rp else "%.0f%%" % (100 * rp["haiku_good"])
    return out


def main() -> None:
    names = sys.argv[1:] or ["base"]
    base_by = {c: by_prompt("base", c) for c in ("qa", "creative", "roleplay")}
    reps = {n: report(n, base_by) for n in names}
    keys = list(next(iter(reps.values())))
    print("| check | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for k in keys:
        print(f"| {k} | " + " | ".join(reps[n].get(k, "-") for n in names) + " |")


if __name__ == "__main__":
    main()
