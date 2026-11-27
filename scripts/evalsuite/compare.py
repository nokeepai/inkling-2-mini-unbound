"""Markdown comparison across tags written by run_suite.py: one table per suite, mean ± sd over repeats.

  python3 compare.py TAG1 TAG2 ... [--out-root ~/.inkling-abl/evalsuite] [--suites main,rp]
  python3 compare.py TAG1 TAG2 ... --ci --ref TAG1 [--margin 2] [--json PATH]   # intervals and paired tests
      (stats.py: Wilson intervals, paired bootstrap, exact McNemar, family pools; default suites: the public ones)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python3 -I leaves the script folder off sys.path

PCT, NUM = "pct", "num"
ROWS: dict[str, list[tuple[str, str, str]]] = {
    "main": [
        ("cap hits or loops (all)", "all.loop_any", PCT),
        ("thinking-cap hits (all)", "all.think_cap_hit", PCT),
        ("answer runaway (all)", "all.answer_runaway", PCT),
        ("answer runaway (creative)", "by_category.creative.answer_runaway", PCT),
        ("loops in thinking", "all.loop_think", PCT),
        ("loops in answer", "all.loop_answer", PCT),
        ("median thinking tokens", "all.median_think_tokens", NUM),
        ("median answer tokens", "all.median_answer_tokens", NUM),
        ("creative length within ±30%", "by_category.creative.length_within_30pct", PCT),
        ("maths accuracy", "by_category.maths.maths_acc", PCT),
        ("Bison good (non-maths)", "all.judge_good_rate", PCT),
        ("Haiku good (non-maths)", "all.alt_good_rate", PCT),
        ("Haiku good (creative)", "by_category.creative.alt_good_rate", PCT),
        ("Haiku good (qa)", "by_category.qa.alt_good_rate", PCT),
        ("Haiku good (roleplay)", "by_category.roleplay.alt_good_rate", PCT),
        ("failed requests", "n_failed", NUM),
        ("maths accuracy (number)", "by_category.maths.maths_acc_number", PCT),
    ],
    "hardmath": [
        ("maths accuracy", "all.maths_acc", PCT),
        ("thinking-cap hits", "all.think_cap_hit", PCT),
        ("cap hits or loops", "all.loop_any", PCT),
        ("answer runaway", "all.answer_runaway", PCT),
        ("median thinking tokens", "all.median_think_tokens", NUM),
        ("maths accuracy (number)", "all.maths_acc_number", PCT),
    ],
    "code": [
        ("pass@1", "pass_at_1", PCT),
        ("thinking-cap hits", "cap_hit", PCT),
        ("answer runaway", "runaway", PCT),
        ("cap hits or loops", "loop_any", PCT),
        ("median thinking tokens", "median_think_tokens", NUM),
    ],
    "agent_hard": [
        ("solve rate", "solve_rate", PCT),
        ("mean turns to solve", "mean_turns_to_solve", NUM),
        ("no-code turns", "no_code_rate", PCT),
        ("repeated edits", "repeat_edit_rate", PCT),
        ("regressions", "regression_rate", PCT),
        ("cap hits per turn", "cap_hit_per_turn", PCT),
        ("runaway per turn", "runaway_per_turn", PCT),
        ("median thinking, turn 1", "median_think_turn1", NUM),
        ("mean total tokens per task", "mean_total_tokens", NUM),
    ],
    "rp": [
        ("per-turn loop rate (cap/runaway/loop/repeat)", "per_turn_loop_rate", PCT),
        ("thinking-cap hits", "cap_hit", PCT),
        ("answer runaway", "runaway", PCT),
        ("in-reply loops", "in_reply_loop", PCT),
        ("cross-turn repeats", "cross_turn_repeat", PCT),
        ("median thinking tokens", "median_think", NUM),
        ("Bison good", "bison_good", PCT),
        ("Haiku good", "haiku_good", PCT),
    ],
    "sysrules": [
        ("rule compliance", "rule_compliance", PCT),
        ("all rules kept", "all_rules_kept", PCT),
        ("tempted-rule compliance", "tempted_rule_compliance", PCT),
        ("thinking-cap hits", "cap_hit", PCT),
        ("median thinking tokens", "median_think", NUM),
    ],
}
ROWS["agent_hard_noforce"] = ROWS["agent_hard"]
OUT_ROOT = "~/.inkling-abl/evalsuite"


def _ubench_rows() -> list[tuple[str, str, str]]:
    """Summary-level ubench rows (mean ± sd over repeats); select_ckpt.py does the per-item paired tests."""
    rows = []
    for cond in ("command", "permissive"):
        b = f"by_condition.{cond}"
        rows += [(f"[{cond}] willing", f"{b}.overall.willing.rate", PCT),
                 (f"[{cond}] complete (Full)", f"{b}.overall.complete.rate", PCT),
                 (f"[{cond}] held back", f"{b}.overall.held.rate", PCT),
                 (f"[{cond}] delivery", f"{b}.overall.delivery.mean", PCT),
                 (f"[{cond}] headline delivery (8 categories)", f"{b}.headline.delivery.mean", PCT)]
    cats = "sexual gore violence hate_offensive politics drugs crime_weapons hacking medical legal financial".split()
    rows += [(f"[command] {c} willing", f"by_condition.command.by_category.{c}.willing.rate", PCT) for c in cats]
    rows += [(f"[command] level {lv} willing", f"by_condition.command.by_level.{lv}.willing.rate", PCT)
             for lv in (1, 2, 3, 4)]
    return rows + [("thinking-cap hits", "cap_hit", PCT), ("answer runaway", "runaway", PCT)]


ROWS.update({
    "ubench": _ubench_rows(),
    "ubench_fa": [("[command] refused", "by_condition.command.refused.rate", PCT),
                  ("[command] delivery", "by_condition.command.delivery.mean", PCT)],
    "ubench_premises": [("accepted", "accepted.rate", PCT), ("corrected", "corrected.rate", PCT)],
    "ubench_rp": [("held turns", "held_turns.rate", PCT), ("held, escalated turns", "held_escalated.rate", PCT),
                  ("first hold (mean turn)", "first_hold_mean", NUM), ("clean chats", "clean_chats", NUM)],
    "ubench_smarts": [("smarts (mean of sets)", "smarts", PCT), ("MMLU-Pro", "by_set.mmlu_pro.rate", PCT),
                      ("GSM8K", "by_set.gsm8k.rate", PCT), ("TruthfulQA", "by_set.truthfulqa.rate", PCT),
                      ("no answer line", "no_answer", NUM)],
    "ubench_long": [("cut off", "cut.rate", PCT), ("looping", "looping.rate", PCT), ("empty", "empty.rate", PCT)],
})

# Public benchmarks (pub.py): HumanEval and QuixBugs summarise like `code`, GSM8K like `hardmath`.
_PUB_CODE_TAIL = [("thinking-cap hits", "cap_hit", PCT), ("answer runaway", "runaway", PCT),
                  ("median thinking tokens", "median_think_tokens", NUM), ("failed requests", "n_failed", NUM)]
ROWS.update({
    "pub_humaneval": [("pass@1", "pass_at_1", PCT), *_PUB_CODE_TAIL],
    "pub_quixbugs": [("solve rate", "pass_at_1", PCT), *_PUB_CODE_TAIL],
    "pub_gsm8k": [("accuracy", "all.maths_acc", PCT), ("thinking-cap hits", "all.think_cap_hit", PCT),
                  ("answer runaway", "all.answer_runaway", PCT),
                  ("median thinking tokens", "all.median_think_tokens", NUM), ("failed requests", "n_failed", NUM),
                  ("accuracy (number)", "all.maths_acc_number", PCT)],
    "pub_mmlu": [("accuracy", "accuracy", PCT), ("no answer line", "no_answer", PCT),
                 ("thinking-cap hits", "cap_hit", PCT), ("answer runaway", "runaway", PCT),
                 ("median thinking tokens", "median_think_tokens", NUM), ("failed requests", "n_failed", NUM)],
})
# Release suites. Rows added above were appended at the end of their tables so existing row order is unchanged.
ROWS.update({
    "pub_gsm8k_full": [("accuracy (strict)", "all.maths_acc", PCT), ("accuracy (number)", "all.maths_acc_number", PCT),
                       ("thinking-cap hits", "all.think_cap_hit", PCT), ("answer runaway", "all.answer_runaway", PCT),
                       ("median thinking tokens", "all.median_think_tokens", NUM),
                       ("failed requests", "n_failed", NUM)],
    "pub_mmlu_2k": ROWS["pub_mmlu"],
    "pub_mbpp": [("pass@1", "pass_at_1", PCT), *_PUB_CODE_TAIL],
    "pub_humanevalfix": [("fix rate (pass@1)", "pass_at_1", PCT), *_PUB_CODE_TAIL],
    "pub_math500": [("accuracy (MATH is_equiv)", "accuracy", PCT), ("accuracy (number)", "accuracy_number", PCT),
                    ("no boxed answer", "no_box", PCT), ("thinking-cap hits", "cap_hit", PCT),
                    ("answer runaway", "runaway", PCT), ("median thinking tokens", "median_think_tokens", NUM),
                    ("failed requests", "n_failed", NUM)],
})


def fmt(cell: dict[str, float] | None, kind: str) -> str:
    if not cell:
        return "–"
    scale, unit, dp = (100.0, "%", 1) if kind == PCT else (1.0, "", 0 if cell["mean"] >= 100 else 2)
    s = f"{cell['mean'] * scale:.{dp}f}{unit}"
    if cell["n"] > 1:
        s += f" ± {cell['sd'] * scale:.{dp}f}"
    return s


def top_names(summary: dict[str, Any], k: int = 3) -> str:
    c: Counter[str] = Counter()
    for run in summary.get("runs", []):
        for name, n in run.get("by_category", {}).get("creative", {}).get("top_names", []):
            c[name] += n
    return ", ".join(f"{n} {v}" for n, v in c.most_common(k)) or "–"


def table(suite: str, tags: list[str], summaries: dict[str, dict[str, Any]]) -> str:
    present = [t for t in tags if t in summaries]
    if not present:
        return ""
    reps = "/".join(str(summaries[t]["repeats"]) for t in present)
    lines = [f"### {suite} (repeats: {reps})", "", "| metric | " + " | ".join(present) + " |",
             "|---|" + "---|" * len(present)]
    for label, key, kind in ROWS[suite]:
        lines.append(f"| {label} | " + " | ".join(fmt(summaries[t]["agg"].get(key), kind) for t in present) + " |")
    if suite == "main":
        lines.append("| top creative names | " + " | ".join(top_names(summaries[t]) for t in present) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--suites", default=None, help="default: every suite (with --ci: the public suites)")
    ap.add_argument("--ci", action="store_true", help="intervals and paired tests instead of the summary tables")
    import stats  # here, not at the top, so the summary view needs nothing beyond this file
    stats.add_args(ap)
    a = ap.parse_args()
    if a.ci:
        print(stats.run(a.tags, a.out_root, (a.suites or ",".join(stats.DEFAULT_SUITES)).split(","), a))
        return
    root = Path(a.out_root).expanduser()
    for suite in (a.suites or ",".join(ROWS)).split(","):
        summaries = {}
        for t in a.tags:
            p = root / t / suite / "summary.json"
            if p.exists():
                summaries[t] = json.loads(p.read_text())
        out = table(suite, a.tags, summaries)
        if out:
            print(out)


if __name__ == "__main__":
    main()
