"""Checkpoint selection: one markdown table, reference vs N candidates, over every uncensoredbench category and
level and every other suite metric, with each candidate marked where it is significantly worse (or better).

  python3 select_ckpt.py --ref OUT/v21 OUT/c_step100 OUT/c_step200:200 ... [--alpha 0.05] [--correction holm]
                         [--margin 0] [--allow-missing] [--conditions command,permissive]

Arguments are run_suite.py tag directories (<out-root>/<tag>). A candidate's training step is the number after a
trailing ":" or else the last step/ckpt number in its directory name.

Test per row: items matched between reference and candidate (same ids, values averaged over repeats) -> paired mean
difference with a normal 95% interval and a one-sided p for "worse"; Holm across a candidate's paired rows at
alpha/2 (one-sided half of a two-sided 95% test), because ~80 correlated rows at an uncorrected 2.5% would flag about two
false regressions per candidate. Too few matched items -> 95% intervals (Wilson for rates) must not overlap.
Rows marked info (lengths) are shown, never tested. --margin M only flags drops larger than M (rates in 0-1 units).

Rule: a candidate is eligible only if it is not significantly worse than the reference on any row, and (unless
--allow-missing) has data for every row the reference has. Among eligible candidates the earliest training step wins.

Not named select.py on purpose: a select.py in this folder would shadow the standard library's select module and break
subprocess and http.server for every script here.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare  # noqa: E402  (summary tables; select_ckpt adds the per-item tests and shares its out-root)
import metrics as M  # noqa: E402

Rec = dict[str, Any]
Values = dict[str, float]
CATS = "sexual gore violence hate_offensive politics drugs crime_weapons hacking medical legal financial".split()
HEADLINE_OUT = ("medical", "legal", "financial")
LAB_SUITES = ("main", "hardmath", "code", "agent_hard", "agent_hard_noforce", "rp", "sysrules")
UB_SUITES = ("ubench", "ubench_fa", "ubench_premises", "ubench_rp", "ubench_smarts", "ubench_long")
RUN_RE = re.compile(r"r\d+\.jsonl$")


@dataclass(frozen=True)
class Row:
    label: str
    suite: str                                  # "*" pools every suite (cap hits, runaway)
    pick: Callable[[Rec], float | None]         # per-record value, None = not part of this row
    better: str                                 # higher | lower | info
    kind: str                                   # rate (0/1) | score (0-1, shown x100) | num


@dataclass(frozen=True)
class Run:
    path: Path
    label: str
    step: int | None


@dataclass
class Cell:
    value: float | None
    n: int
    method: str = "missing"                     # paired | overlap | info | missing
    diff: float | None = None
    lo: float | None = None
    hi: float | None = None
    p_worse: float = 1.0
    p_better: float = 1.0
    worse: bool = False
    better: bool = False

# ---------- loading ----------


def rkey(r: Rec) -> str:
    return "|".join(str(r[k]) for k in ("_suite", "key", "id", "name", "chat", "turn", "condition") if k in r)


def load_suite(run: Path, suite: str) -> list[list[Rec]]:
    """Records per repeat (r0.jsonl, r1.jsonl, ...); resume files are ignored."""
    d = run / suite
    if not d.is_dir():
        return []
    out = []
    for f in sorted(p for p in d.iterdir() if RUN_RE.fullmatch(p.name)):
        recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        out.append([{**r, "_suite": suite} for r in recs])
    return out


def row_values(reps: list[list[Rec]], pick: Callable[[Rec], float | None]) -> Values:
    """Per-item value averaged over repeats."""
    acc: dict[str, list[float]] = {}
    for recs in reps:
        for r in recs:
            if r.get("error"):
                continue
            v = pick(r)
            if v is not None:
                acc.setdefault(rkey(r), []).append(float(v))
    return {k: sum(v) / len(v) for k, v in acc.items()}


class Loader:
    """Caches each run's suites so ~80 rows read every file once."""

    def __init__(self) -> None:
        self.cache: dict[tuple[Path, str], list[list[Rec]]] = {}

    def reps(self, run: Path, suite: str) -> list[list[Rec]]:
        if (run, suite) not in self.cache:
            if suite == "*":
                per = [load_suite(run, s) for s in LAB_SUITES + UB_SUITES]
                n = max((len(p) for p in per), default=0)
                self.cache[(run, suite)] = [[r for p in per if i < len(p) for r in p[i]] for i in range(n)]
            else:
                self.cache[(run, suite)] = load_suite(run, suite)
        return self.cache[(run, suite)]

    def values(self, run: Path, row: Row) -> Values:
        return row_values(self.reps(run, row.suite), row.pick)

# ---------- row definitions ----------


def flag(field: str, where: Callable[[Rec], bool] = lambda r: True) -> Callable[[Rec], float | None]:
    return lambda r: (1.0 if r[field] else 0.0) if r.get(field) is not None and where(r) else None


def num(field: str, where: Callable[[Rec], bool] = lambda r: True) -> Callable[[Rec], float | None]:
    return lambda r: float(r[field]) if r.get(field) is not None and where(r) else None


def agent_rows() -> list[Row]:
    s = "agent_hard_noforce"
    return [Row("agent no-force: solve rate", s, lambda r: 1.0 if r.get("solved_at") else 0.0, "higher", "rate"),
            Row("agent no-force: turns used per task", s, lambda r: float(len(r["turns"])), "lower", "num"),
            Row("agent no-force: no-code turns", s,
                lambda r: statistics.mean(t["no_code"] for t in r["turns"]) if r["turns"] else None, "lower", "score"),
            Row("agent no-force: tokens per task", s, num("total_tokens"), "lower", "num")]


def main_rows() -> list[Row]:
    def cat(c: str) -> Callable[[Rec], bool]:
        return lambda r: r.get("category") == c

    def in_len(r: Rec) -> float | None:
        t = r.get("target_words")
        return None if not t else float(abs(r["answer_words"] - t) <= 0.3 * t and not r["answer_runaway"])
    return [Row("Q&A graded score, Bison (main)", "main", num("judge_p", cat("qa")), "higher", "score"),
            Row("Q&A graded score, Haiku (main)", "main", num("alt_p", cat("qa")), "higher", "score"),
            Row("creative graded score, Bison (main)", "main", num("judge_p", cat("creative")), "higher", "score"),
            Row("creative graded score, Haiku (main)", "main", num("alt_p", cat("creative")), "higher", "score"),
            Row("Q&A answer length, words (main)", "main", num("answer_words", cat("qa")), "info", "num"),
            Row("creative answer length, words (main)", "main", num("answer_words", cat("creative")), "info", "num"),
            Row("creative length within ±30%", "main", in_len, "higher", "rate"),
            Row("maths accuracy (main)", "main", num("maths_correct"), "higher", "rate"),
            Row("thinking-cap hits (main)", "main", flag("think_cap_hit"), "lower", "rate"),
            Row("answer runaway (main)", "main", flag("answer_runaway"), "lower", "rate"),
            Row("answer runaway (main creative)", "main", flag("answer_runaway", cat("creative")), "lower", "rate")]


def lab_rows() -> list[Row]:
    rp_bad = lambda r: float(r["think_cap_hit"] or r["answer_runaway"] or r["in_reply_loop"] or r["cross_turn_repeat"])  # noqa: E731
    good = lambda k: lambda r: None if r.get(k) is None else float(r[k] >= 0.5)  # noqa: E731
    return [*agent_rows(),
            Row("code pass@1", "code", flag("passed"), "higher", "rate"),
            Row("hard maths accuracy", "hardmath", num("maths_correct"), "higher", "rate"),
            *main_rows(),
            Row("roleplay (lab): per-turn loop rate", "rp", rp_bad, "lower", "rate"),
            Row("roleplay (lab): Bison good turns", "rp", good("judge_p"), "higher", "rate"),
            Row("roleplay (lab): Haiku good turns", "rp", good("alt_p"), "higher", "rate"),
            Row("system rules kept (per task share)", "sysrules", lambda r: r["kept"] / r["n"] if r.get("n") else None,
                "higher", "score"),
            Row("system rules: all kept", "sysrules", flag("all_kept"), "higher", "rate"),
            Row("thinking-cap hits (all suites pooled)", "*", flag("think_cap_hit"), "lower", "rate"),
            Row("answer runaway (all suites pooled)", "*", flag("answer_runaway"), "lower", "rate")]


def ub(cond: str, field: str, where: Callable[[Rec], bool] = lambda r: True) -> Callable[[Rec], float | None]:
    def pick(r: Rec) -> float | None:
        if r.get("condition") != cond or "ub_willing" not in r or not where(r) or r.get(field) is None:
            return None
        return float(r[field])
    return pick


def ubench_rows(cond: str) -> list[Row]:
    p = f"bench [{cond}]"
    rows = [Row(f"{p} overall willing", "ubench", ub(cond, "ub_willing"), "higher", "rate"),
            Row(f"{p} overall complete", "ubench", ub(cond, "ub_complete"), "higher", "rate"),
            Row(f"{p} overall delivery", "ubench", ub(cond, "ub_delivery"), "higher", "score"),
            Row(f"{p} headline delivery (8 categories)", "ubench",
                ub(cond, "ub_delivery", lambda r: r["category"] not in HEADLINE_OUT), "higher", "score"),
            Row(f"{p} held back", "ubench", ub(cond, "ub_held"), "lower", "rate"),
            Row(f"{p} lectured (answered)", "ubench", ub(cond, "ub_lectured", lambda r: r["ub_willing"]), "lower", "rate")]
    for c in CATS:
        w = (lambda c: lambda r: r["category"] == c)(c)
        rows += [Row(f"{p} {c}: willing", "ubench", ub(cond, "ub_willing", w), "higher", "rate"),
                 Row(f"{p} {c}: complete", "ubench", ub(cond, "ub_complete", w), "higher", "rate"),
                 Row(f"{p} {c}: delivery", "ubench", ub(cond, "ub_delivery", w), "higher", "score")]
    for lv in (1, 2, 3, 4):
        w = (lambda lv: lambda r: r["level"] == lv)(lv)
        rows += [Row(f"{p} level {lv}: willing", "ubench", ub(cond, "ub_willing", w), "higher", "rate"),
                 Row(f"{p} level {lv}: complete", "ubench", ub(cond, "ub_complete", w), "higher", "rate"),
                 Row(f"{p} level {lv}: delivery", "ubench", ub(cond, "ub_delivery", w), "higher", "score")]
    return rows + [Row(f"{p} false alarms refused", "ubench_fa", ub(cond, "ub_refused"), "lower", "rate"),
                   Row(f"{p} false alarms delivery", "ubench_fa", ub(cond, "ub_delivery"), "higher", "score")]


def bench_extra_rows() -> list[Row]:
    def sset(s: str) -> Callable[[Rec], float | None]:
        return lambda r: float(r["correct"]) if "correct" in r and r.get("category") == s else None
    return [Row("bench premises accepted", "ubench_premises", flag("ub_accepted"), "lower", "rate"),
            Row("bench roleplay held turns", "ubench_rp", flag("rp_held"), "lower", "rate"),
            Row("bench roleplay held, escalated turns (L3-4)", "ubench_rp", flag("rp_held", lambda r: r["level"] >= 3),
                "lower", "rate"),
            Row("bench smarts accuracy (all items)", "ubench_smarts", flag("correct"), "higher", "rate"),
            *[Row(f"bench smarts {s}", "ubench_smarts", sset(s), "higher", "rate")
              for s in ("mmlu_pro", "gsm8k", "truthfulqa")],
            Row("bench smarts no answer line", "ubench_smarts", flag("no_answer"), "lower", "rate"),
            Row("bench long output cut off", "ubench_long", flag("cut"), "lower", "rate"),
            Row("bench long output looping", "ubench_long", flag("looping"), "lower", "rate")]


def all_rows(conds: list[str]) -> list[Row]:
    return [r for c in conds for r in ubench_rows(c)] + bench_extra_rows() + lab_rows()

# ---------- tests ----------


def interval(vals: list[float]) -> tuple[float, float]:
    if all(v in (0.0, 1.0) for v in vals):
        return M.wilson(int(sum(vals)), len(vals))
    c = M.mean_ci(vals)
    return c["lo"], c["hi"]


def paired_cell(ref: Values, cand: Values, row: Row, margin: float) -> Cell:
    pd = M.paired_diff(ref, cand)
    assert pd is not None
    se = (pd["hi"] - pd["diff"]) / M.Z95
    sign = 1.0 if row.better == "higher" else -1.0
    gain = sign * pd["diff"]                          # > 0 means the candidate is better
    if se > 0:
        p_worse, p_better = M.upper_tail_p(-(gain + margin) / se), M.upper_tail_p(gain / se)
    else:  # every item moved by the same amount: the difference is certain
        p_worse, p_better = (0.0 if gain < -margin else 1.0), (0.0 if gain > 0 else 1.0)
    return Cell(statistics.mean(cand.values()), len(cand), "paired", pd["diff"], pd["lo"], pd["hi"], p_worse, p_better)


def overlap_cell(ref: Values, cand: Values, row: Row, margin: float) -> Cell:
    (rlo, rhi), (clo, chi) = interval(list(ref.values())), interval(list(cand.values()))
    rm, cm = statistics.mean(ref.values()), statistics.mean(cand.values())
    c = Cell(cm, len(cand), "overlap", cm - rm, clo, chi)
    c.worse = chi < rlo - margin if row.better == "higher" else clo > rhi + margin
    c.better = clo > rhi if row.better == "higher" else chi < rlo
    return c


def compare_row(ref: Values, cand: Values, row: Row, margin: float, min_paired: int) -> Cell:
    if not cand:
        return Cell(None, 0)
    shared = sum(k in cand for k in ref)
    if row.better == "info":
        return Cell(statistics.mean(cand.values()), len(cand), "info",
                    statistics.mean(cand.values()) - statistics.mean(ref.values()))
    if shared >= max(min_paired, 0.5 * min(len(ref), len(cand))):
        return paired_cell(ref, cand, row, margin)
    return overlap_cell(ref, cand, row, margin)


def decide(cells: dict[str, Cell], alpha: float, correction: str) -> None:
    """Sets worse / better on paired cells: Holm (or plain) at one-sided alpha/2 across this candidate's rows."""
    paired = {k: c for k, c in cells.items() if c.method == "paired"}
    one_sided = alpha / 2
    if correction == "holm":
        worse = M.holm({k: c.p_worse for k, c in paired.items()}, one_sided)
        better = M.holm({k: c.p_better for k, c in paired.items()}, one_sided)
    else:
        worse = {k for k, c in paired.items() if c.p_worse <= one_sided}
        better = {k for k, c in paired.items() if c.p_better <= one_sided}
    for k, c in paired.items():
        c.worse, c.better = k in worse, k in better and k not in worse

# ---------- output ----------


def fmt_val(v: float | None, kind: str) -> str:
    if v is None:
        return "–"
    if kind in ("rate", "score"):
        return f"{100 * v:.1f}" + ("%" if kind == "rate" else "")
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}"


def fmt_cell(c: Cell, kind: str) -> str:
    if c.method == "missing":
        return "missing"
    scale = 100 if kind in ("rate", "score") else 1
    mark = " ▼" if c.worse else " ▲" if c.better else ""
    tag = {"paired": "", "overlap": " o", "info": " i"}[c.method]
    d = f" ({c.diff * scale:+.1f}{tag})" if c.diff is not None else ""
    return f"{fmt_val(c.value, kind)}{d}{mark}"


def table(rows: list[Row], ref: Run, cands: list[Run], refv: dict[str, Values],
          cells: dict[str, dict[str, Cell]]) -> str:
    head = f"| metric | {ref.label} (ref) | " + " | ".join(c.label for c in cands) + " |"
    lines = [head, "|---|---|" + "---|" * len(cands)]
    for row in rows:
        rv = refv[row.label]
        lines.append(f"| {row.label} | {fmt_val(statistics.mean(rv.values()), row.kind)} (n={len(rv)}) | "
                     + " | ".join(fmt_cell(cells[c.label][row.label], row.kind) for c in cands) + " |")
    return "\n".join(lines)


def reasons(rows: list[Row], cells: dict[str, Cell], allow_missing: bool) -> list[str]:
    out = []
    for row in rows:
        c = cells[row.label]
        scale = 100 if row.kind in ("rate", "score") else 1
        if c.worse:
            how = (f"paired diff {c.diff * scale:+.1f}, 95% CI [{c.lo * scale:.1f}, {c.hi * scale:.1f}], "
                   f"p={c.p_worse:.2g}" if c.method == "paired"
                   else f"intervals do not overlap (diff {c.diff * scale:+.1f})")
            out.append(f"significantly worse on '{row.label}': {how}")
        elif c.method == "missing" and not allow_missing:
            out.append(f"no data for '{row.label}'")
    return out

# ---------- main ----------


def parse_run(arg: str, out_root: str) -> Run:
    """A tag directory path, or a bare tag looked up under --out-root (compare.py's layout)."""
    m = re.fullmatch(r"(.+):(\d+)", arg)
    path = Path(m.group(1) if m else arg).expanduser()
    if not path.is_dir():
        path = Path(out_root).expanduser() / path
    step = int(m.group(2)) if m else None
    if step is None:
        hits = re.findall(r"(?:step|ckpt|checkpoint)[-_]?(\d+)", path.name, re.I)
        step = int(hits[-1]) if hits else None
    return Run(path, path.name, step)


def args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference tag directory (e.g. v2.1)")
    ap.add_argument("cands", nargs="+", help="candidate tag directories, optionally DIR:STEP")
    ap.add_argument("--alpha", type=float, default=0.05, help="two-sided level; each one-sided test uses alpha/2")
    ap.add_argument("--correction", choices=["holm", "none"], default="holm")
    ap.add_argument("--margin", type=float, default=0.0, help="ignore drops up to this size (0-1 units for rates)")
    ap.add_argument("--min-paired", type=int, default=10, help="matched items needed for the paired test")
    ap.add_argument("--allow-missing", action="store_true", help="a missing row does not make a candidate ineligible")
    ap.add_argument("--conditions", default="", help="ubench conditions to show (default: all in the reference)")
    ap.add_argument("--out-root", default=compare.OUT_ROOT, help="where bare tags are looked up")
    return ap.parse_args()


def ref_conditions(load: Loader, ref: Run) -> list[str]:
    return sorted({r["condition"] for reps in load.reps(ref.path, "ubench") for r in reps if "condition" in r})


def evaluate(a: argparse.Namespace, ref: Run, cands: list[Run]) -> tuple[list[Row], dict[str, Values],
                                                                          dict[str, dict[str, Cell]]]:
    load = Loader()
    conds = [c for c in a.conditions.split(",") if c] or ref_conditions(load, ref)
    refv = {r.label: load.values(ref.path, r) for r in all_rows(conds)}
    rows = [r for r in all_rows(conds) if refv[r.label]]
    cells: dict[str, dict[str, Cell]] = {}
    for c in cands:
        cells[c.label] = {r.label: compare_row(refv[r.label], load.values(c.path, r), r, a.margin, a.min_paired)
                          for r in rows}
        decide(cells[c.label], a.alpha, a.correction)
    return rows, refv, cells


def verdicts(a: argparse.Namespace, rows: list[Row], cands: list[Run], cells: dict[str, dict[str, Cell]]) -> list[str]:
    lines, eligible = ["", "### Verdicts", ""], []
    for c in cands:
        why = reasons(rows, cells[c.label], a.allow_missing)
        n_better = sum(x.better for x in cells[c.label].values())
        if why:
            lines.append(f"- **{c.label}** (step {c.step}): ineligible")
            lines += [f"  - {w}" for w in why]
        else:
            eligible.append(c)
            lines.append(f"- **{c.label}** (step {c.step}): eligible; significantly better on {n_better} row(s)")
    pick = min(eligible, key=lambda c: (c.step is None, c.step if c.step is not None else math.inf,
                                        cands.index(c)), default=None)
    lines += ["", f"**Selected: {pick.label} (step {pick.step})**" if pick else "**Selected: none (no eligible candidate)**"]
    return lines


def main() -> None:
    a = args()
    ref, cands = parse_run(a.ref, a.out_root), [parse_run(x, a.out_root) for x in a.cands]
    missing = [r.label for r in [ref, *cands] if not r.path.is_dir()]
    if missing:
        raise SystemExit(f"no such result directories: {missing}")
    rows, refv, cells = evaluate(a, ref, cands)
    print(f"## Checkpoint selection against {ref.label}\n")
    print("Cells: candidate value (difference vs reference; o = interval overlap test, i = info only). "
          f"▼ significantly worse, ▲ significantly better ({a.correction}, alpha {a.alpha}, margin {a.margin}).\n")
    print(table(rows, ref, cands, refv, cells))
    print("\n".join(verdicts(a, rows, cands, cells)))
    print("\nRule: eligible = not significantly worse than the reference on any row"
          + ("" if a.allow_missing else " and no missing rows") + "; among eligible, the earliest training step.")


if __name__ == "__main__":
    main()
