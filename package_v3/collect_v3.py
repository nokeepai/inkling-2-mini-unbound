#!/usr/bin/env python3
"""Fill package_v3/values.json from graded eval results on disk. Prints numbers only, never model text.

  python3 -I package_v3/collect_v3.py --root ~/.inkling-abl/stack_evals/judged \
      --stock stock --prime PRIME_TAG --v21 v21 --v3 V3_TAG \
      --indep-prime CI_PRIME.json --indep-v3 CI_V3.json --margin 2 \
      [--root-e03 ~/.inkling-abl/stack_evals/judged_e03 --e03-stock stock --e03-v21 v21] \
      [--gguf-table gguf_table.tsv] [--dry-run] [--out PATH]

Sources, and the one place each is read:
- <root>/<tag>/<suite>/summary.json: lab aggregates, looked up by scripts/evalsuite/compare.py's row labels.
- <root>/<tag>/<suite>/r0.jsonl: per-item scores through scripts/evalsuite/stats.py (the same scorers, Wilson
  intervals and paired bootstrap as compare.py --ci), plus thinking-token counts and seconds. Answer text is only
  touched inside stats.py's maths "number" scorer, in memory; nothing from it is printed or written.
- --indep-prime / --indep-v3: compare.py TAGS --ci --ref BASE --json PATH output (indep_rows(), indep_families()).
- --root-e03: the same suites at the low effort setting, for the "why not just lower effort" table.
- --gguf-table: the TSV written by scripts/v3stack/gguf_eval_all.sh (gguf_eval_lib.py table).

A value whose input is missing is never invented: a provisional value stays provisional (render --final still
refuses it), anything else becomes "TBD". Every such case is printed.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts" / "evalsuite"))
import compare  # noqa: E402  (row keys: the same metric definitions as the lab's comparison tables)
import stats as ST  # noqa: E402  (per-item scorers, Wilson, paired bootstrap, verdict rule)

MODELS = ("stock", "prime", "v21", "v3")
PAIRS = {"prime": ("stock", "prime"), "v3": ("v21", "v3")}
PAIR_GROUP = {"prime": "prime_vs_stock", "v3": "v3_vs_v21"}
Leaf = dict
Found = dict[str, Union[Leaf, None]]
Row = dict[str, object]
RESAMPLES, SEED = 10000, 1234   # compare.py --ci defaults, so lab intervals match the published JSON's method

# Public benchmarks: value name -> (suites, larger first; stats.py metric). The first suite present wins, so the
# full runs replace the lab-size ones without a flag.
BENCH: dict[str, tuple[tuple[str, ...], str]] = {
    "humaneval": (("pub_humaneval",), "pass@1"),
    "mbpp": (("pub_mbpp",), "pass@1"),
    "humanevalfix": (("pub_humanevalfix",), "pass@1"),
    "quixbugs": (("pub_quixbugs",), "pass@1"),
    "gsm8k_strict": (("pub_gsm8k_full", "pub_gsm8k"), "strict"),
    "gsm8k_lenient": (("pub_gsm8k_full", "pub_gsm8k"), "number"),
    "math500": (("pub_math500",), "is_equiv"),
    "mmlu": (("pub_mmlu_2k", "pub_mmlu"), "accuracy"),
}
# MATH-500's "number correct" reading is not in stats.METRICS; it is the row's own correct_number flag (pub.py).
MATH500_NUMBER = "math500_number"
SOURCE_TEXT = {
    "pub_humaneval": "openai/human-eval, all {n} problems", "pub_mbpp": "MBPP sanitized, test split, {n} problems",
    "pub_humanevalfix": "HumanEvalFix (HumanEvalPack, Python), {n} problems",
    "pub_quixbugs": "QuixBugs, {n} Python programs with JSON test cases",
    "pub_gsm8k_full": "GSM8K test, all {n} problems", "pub_gsm8k": "GSM8K test, seeded sample of {n} problems",
    "pub_math500": "MATH-500, {n} problems",
    "pub_mmlu_2k": "MMLU test, {n} questions stratified by subject",
    "pub_mmlu": "MMLU test, {n} questions stratified by subject (lab sample)",
}
FAMILY_KEY = {("coding", "pass@1"): "fam_coding", ("bug-fixing", "pass@1"): "fam_bugfix",
              ("maths", "strict"): "fam_maths_strict", ("maths", "number"): "fam_maths_number",
              ("knowledge", "accuracy"): "fam_knowledge"}
SUITE_SHORT = {"pub_humaneval": "HumanEval", "pub_mbpp": "MBPP", "pub_humanevalfix": "HumanEvalFix",
               "pub_quixbugs": "QuixBugs", "pub_gsm8k_full": "GSM8K", "pub_gsm8k": "GSM8K (sample)",
               "pub_math500": "MATH-500", "pub_mmlu_2k": "MMLU 2,000", "pub_mmlu": "MMLU 400-question lab sample"}
GGUF_TAGS = ("prime_q8", "prime_q4km", "prime_iq2m", "stock_q4km")
GGUF_ROWS = {"solve": "agent_hard_noforce.solve_rate", "nocode": "agent_hard_noforce.no_code_rate",
             "tokens": "agent_hard_noforce.mean_total_tokens", "turns": "agent_hard_noforce.turns_total",
             "gsm8k_number": "pub_gsm8k.all.maths_acc_number", "humaneval": "pub_humaneval.pass_at_1"}


# ---- reading -------------------------------------------------------------------
@dataclass
class Source:
    root: Path
    tags: dict[str, str]  # model -> tag
    missing: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        if text not in self.missing:
            self.missing.append(text)

    def suite_dir(self, model: str, suite: str) -> Path | None:
        d = self.root / self.tags[model] / suite
        if (d / "summary.json").exists():
            return d
        self.note(f"{suite} missing for {model} (tag {self.tags[model]} under {self.root})")
        return None

    def agg(self, model: str, suite: str, key: str) -> float | None:
        d = self.suite_dir(model, suite)
        if d is None:
            return None
        cell = json.loads((d / "summary.json").read_text())["agg"].get(key)
        return float(cell["mean"]) if cell else None

    def rows(self, model: str, suite: str) -> list[Row] | None:
        d = self.suite_dir(model, suite)
        if d is None or not (d / "r0.jsonl").exists():
            return None
        return [json.loads(x) for x in (d / "r0.jsonl").read_text().splitlines() if x.strip()]

    def items(self, model: str, suite: str) -> ST.Items | None:
        it = ST.load_items(self.root, self.tags[model], suite, 0)
        if it is None:
            self.note(f"{suite} missing for {model} (tag {self.tags[model]} under {self.root})")
        return it

    def has(self, model: str, suite: str) -> bool:
        return (self.root / self.tags[model] / suite / "r0.jsonl").exists()

    def meta(self, model: str, suite: str) -> dict | None:
        p = self.root / self.tags[model] / f"meta_{suite}.json"
        if not p.exists():   # one-run roots keep only meta.json (run_suite.py rewrites it per call)
            p = self.root / self.tags[model] / "meta.json"
        return json.loads(p.read_text()) if p.exists() else None


def row_key(suite: str, label: str) -> str:
    """Look the summary key up by its compare.py row label, so a renamed metric fails loudly here."""
    for lab, key, _ in compare.ROWS[suite]:
        if lab == label:
            return key
    raise KeyError(f"compare.ROWS[{suite!r}] has no row {label!r}")


def outcome(suite: str, metric: str) -> ST.Outcome:
    if metric == MATH500_NUMBER:
        return lambda r: bool(r.get("correct_number"))
    return ST.metric_fn(suite, metric)


# ---- leaves ----------------------------------------------------------------------
def leaf(value: float | int | str, unit: str, source: str, ci: tuple[float, float] | None = None,
         display: str | None = None) -> Leaf:
    out: Leaf = {"value": round(value, 4) if isinstance(value, float) else value, "unit": unit, "source": source}
    if ci is not None:
        out["ci"] = [round(ci[0], 2), round(ci[1], 2)]
    if display:
        out["display"] = display
    return out


def pct(x: float | None, source: str, ci: tuple[float, float] | None = None) -> Leaf | None:
    return None if x is None else leaf(100.0 * x, "%", source, ci=None if ci is None else (100 * ci[0], 100 * ci[1]))


def num(x: float | None, unit: str, source: str) -> Leaf | None:
    return None if x is None else leaf(round(x, 1), unit, source)


def n_leaf(ns: list[int], source: str) -> Leaf | None:
    """One n when every model answered the same count; otherwise show each, never pick one."""
    if not ns:
        return None
    if len(set(ns)) == 1:
        return leaf(ns[0], "n", source)
    return leaf(min(ns), "n", source, display=" / ".join(f"{x:,}" for x in ns))


def score_leaf(cell: dict, source: str) -> Leaf:
    """A stats.single() / compare --json row: score with its Wilson interval."""
    return leaf(100 * cell["score"], "%", source, ci=(100 * cell["wilson_lo"], 100 * cell["wilson_hi"]))


VERDICT_DISPLAY = {
    "WORSE": "worse (whole interval below zero)",
    "non-inferior, better": "non-inferior (interval above zero; no gain claimed)",
    "non-inferior": "non-inferior",
    "inconclusive": "inconclusive (interval reaches past the margin)",
}


def verdict_leaf(lo: float, hi: float, margin: float | None, source: str) -> Leaf:
    """lo/hi as fractions. stats.verdict is the rule compare.py prints, so card and lab report always agree."""
    if margin is None:
        return leaf("margin TBD", "text", source)
    raw = ST.verdict(lo, hi, margin / 100)
    return leaf(raw, "text", source, display=VERDICT_DISPLAY.get(raw, raw))


def paired_leaves(prefix: str, base: dict, fixed: dict, vs: dict, margin: float | None, source: str) -> Found:
    """base/fixed: stats.single() cells; vs: stats.paired() cell. Writes <prefix>_{base,fixed,diff,verdict,n}."""
    ns = [int(base["n"]), int(fixed["n"])]
    n = leaf(int(vs["n_paired"]), "n", source,
             display=None if len(set(ns + [int(vs["n_paired"])])) == 1 else f"{int(vs['n_paired']):,} paired")
    return {f"{prefix}_base": score_leaf(base, source), f"{prefix}_fixed": score_leaf(fixed, source),
            f"{prefix}_diff": leaf(100 * vs["diff"], "pts", source, ci=(100 * vs["boot_lo"], 100 * vs["boot_hi"])),
            f"{prefix}_verdict": verdict_leaf(vs["boot_lo"], vs["boot_hi"], margin, source), f"{prefix}_n": n}


def empty(prefix: str, parts: tuple[str, ...] = ("base", "fixed", "diff", "verdict", "n")) -> Found:
    return {f"{prefix}_{p}": None for p in parts}


def paired_from_items(src: Source, pair: str, suite: str, metric: str, margin: float | None,
                      prefix: str) -> Found:
    b, f = PAIRS[pair]
    ib, ifx = src.items(b, suite), src.items(f, suite)
    if ib is None or ifx is None:
        return empty(prefix)
    fn = outcome(suite, metric)
    sb, sf = ST.scored(ib, fn), ST.scored(ifx, fn)
    if not sb or not sf:
        return empty(prefix)
    vs = ST.paired(sb, sf, (margin or 2.0) / 100, RESAMPLES, SEED)
    return paired_leaves(prefix, ST.single(sb), ST.single(sf), vs, margin, f"{suite} {metric} per item")


# ---- per-pair lab checks ------------------------------------------------------------
Getter = Callable[[str], Union[Leaf, None]]


def lab_getters(src: Source) -> dict[str, Getter]:
    """value-key prefix -> function(model) -> leaf. Labels are compare.py's, so definitions cannot drift."""
    def agg_pct(suite: str, label: str) -> Getter:
        return lambda m: pct(src.agg(m, suite, row_key(suite, label)), f"{suite}:{src.tags[m]}")

    def agg_num(suite: str, label: str, unit: str) -> Getter:
        return lambda m: num(src.agg(m, suite, row_key(suite, label)), unit, f"{suite}:{src.tags[m]}")

    def agg_wilson(suite: str, label: str, n_key: str) -> Getter:
        """A rate over independent items (tasks, judged answers): Wilson 95% interval from its own n."""
        def get(m: str) -> Leaf | None:
            rate, n = src.agg(m, suite, row_key(suite, label)), src.agg(m, suite, n_key)
            if rate is None or not n:
                return pct(rate, f"{suite}:{src.tags[m]}")
            return pct(rate, f"{suite}:{src.tags[m]}", ST.M.wilson(round(rate * n), int(n)))
        return get

    return {
        "debug_solved": agg_wilson("agent_hard_noforce", "solve rate", "tasks"),
        # Turns within one task are not independent, so no-code turns get no interval (n turns is shown instead).
        "debug_nocode": agg_pct("agent_hard_noforce", "no-code turns"),
        "debug_tokens": agg_num("agent_hard_noforce", "mean total tokens per task", "tokens"),
        "debug_turns": lambda m: num(src.agg(m, "agent_hard_noforce", "turns_total"), "n", src.tags[m]),
        "quality_good": agg_wilson("main", "Haiku good (non-maths)", "all.alt_n"),
        "creative_good": agg_wilson("main", "Haiku good (creative)", "by_category.creative.alt_n"),
        "answer_tokens": agg_num("main", "median answer tokens", "tokens"),
        "rp_good": agg_pct("rp", "Haiku good"),   # turns of one chat are not independent: no interval
    }


def sample_size(src: Source, pair: str, suite: str, key: str) -> Leaf | None:
    vals = [src.agg(m, suite, key) for m in PAIRS[pair]]
    if any(v is None for v in vals):
        return None
    return n_leaf([int(v) for v in vals if v is not None], suite)


def lab_intervals(src: Source, pair: str, margin: float | None) -> Found:
    """Lab code and hard maths, scored per item like compare.py --ci: both maths readings, always together."""
    g, (b, f) = PAIR_GROUP[pair], PAIRS[pair]
    out: Found = {}
    for stem, suite, metric in (("code_pass", "code", "pass@1"), ("maths_strict", "hardmath", "strict"),
                                ("maths_lenient", "hardmath", "number")):
        cells = paired_from_items(src, pair, suite, metric, margin, f"{g}.{stem}")
        out[f"{g}.{stem}_{b}"] = cells.pop(f"{g}.{stem}_base")
        out[f"{g}.{stem}_{f}"] = cells.pop(f"{g}.{stem}_fixed")
        out[f"{g}.{stem}_diff"] = cells[f"{g}.{stem}_diff"]
        out[f"{g}.{stem}_verdict"] = cells[f"{g}.{stem}_verdict"]
    return out


def collect_pair(src: Source, pair: str, margin: float | None) -> Found:
    g = PAIR_GROUP[pair]
    out: Found = {}
    for prefix, get in lab_getters(src).items():
        for m in PAIRS[pair]:
            out[f"{g}.{prefix}_{m}"] = get(m)
    for key, suite, agg_key in (("debug_n_tasks", "agent_hard_noforce", "tasks"), ("quality_n", "main", "all.alt_n"),
                                ("creative_n", "main", "by_category.creative.alt_n"), ("rp_n", "rp", "turns"),
                                ("code_n", "code", "n"), ("maths_n", "hardmath", "all.n")):
        out[f"{g}.{key}"] = sample_size(src, pair, suite, agg_key)
    out.update(lab_intervals(src, pair, margin))
    return out


# ---- speed, thinking distribution -----------------------------------------------------
def mean_of(rows: list[Row] | None, pick: Callable[[Row], float]) -> float | None:
    return statistics.fmean(pick(r) for r in rows) if rows else None


def percentile(xs: list[float], q: float) -> float:
    """Nearest-rank percentile: a value that actually occurred, which reads better than an interpolated one."""
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))]


def fnum(r: Row, key: str) -> float:
    v = r.get(key)
    return float(v) if isinstance(v, (int, float)) else 0.0


def collect_speed(src: Source) -> Found:
    out: Found = {}
    for m in MODELS:
        main, ub = src.rows(m, "main"), src.rows(m, "ubench")
        t = src.tags[m]
        out[f"speed.think_main_{m}"] = num(src.agg(m, "main", row_key("main", "median thinking tokens")), "tokens", t)
        out[f"speed.think_ubench_{m}"] = num(src.agg(m, "ubench", "median_think_tokens"), "tokens", t)
        out[f"speed.sec_main_{m}"] = num(mean_of(main, lambda r: fnum(r, "seconds")), "s", t)
        out[f"speed.sec_ubench_{m}"] = num(mean_of(ub, lambda r: fnum(r, "seconds")), "s", t)
        out[f"speed.tokens_answer_{m}"] = num(mean_of(main, lambda r: fnum(r, "think_tokens")
                                                      + fnum(r, "answer_tokens")), "tokens", t)
        out[f"speed.cap_ubench_{m}"] = pct(src.agg(m, "ubench", row_key("ubench", "thinking-cap hits")), t)
        out[f"speed.cap_main_{m}"] = pct(src.agg(m, "main", row_key("main", "cap hits or loops (all)")), t)
    for m in ("v21", "v3"):
        out[f"speed.runaway_ubench_{m}"] = pct(src.agg(m, "ubench", row_key("ubench", "answer runaway")), src.tags[m])
    ns = [src.agg(m, "main", "all.n") for m in MODELS]
    out["speed.main_n"] = n_leaf([int(x) for x in ns if x is not None], "main") if None not in ns else None
    return out


def think_stats(rows: list[Row], model: str, tag: str) -> Found:
    xs = [fnum(r, "think_tokens") for r in rows]
    nonterm = sum(bool(r.get("think_cap_hit") or r.get("answer_runaway")) for r in rows) / len(rows)
    return {f"think_dist.mean_{model}": leaf(float(round(statistics.fmean(xs))), "tokens", tag),
            f"think_dist.median_{model}": leaf(float(statistics.median(xs)), "tokens", tag),
            f"think_dist.p90_{model}": leaf(percentile(xs, 90), "tokens", tag),
            f"think_dist.p99_{model}": leaf(percentile(xs, 99), "tokens", tag),
            f"think_dist.nonterm_{model}": leaf(100.0 * nonterm, "%", tag)}


def collect_think_dist(src: Source) -> Found:
    out: Found = {}
    ns = []
    for m in MODELS:
        rows = src.rows(m, "main")
        if rows:
            out.update(think_stats(rows, m, src.tags[m]))
            ns.append(len(rows))
        else:
            out.update({f"think_dist.{k}_{m}": None for k in ("mean", "median", "p90", "p99", "nonterm")})
    out["think_dist.n"] = n_leaf(ns, "main") if len(ns) == len(MODELS) else None
    out["think_dist.suite"] = leaf("the lab's main set of mixed everyday prompts", "text", "main")
    return out


# ---- decoding settings (what was measured is what the cards state and recommend) ---------
SETTING_KEYS = (("effort", "effort"), ("temperature", "temperature"), ("top_p", "top_p"),
                ("think_cap", "think_cap"), ("answer_cap", "answer_cap"))


def settings_of(src: Source, model: str, suite: str) -> dict[str, object] | None:
    meta = src.meta(model, suite)
    if meta is None:
        return None
    c, a = meta["client_cfg"], meta.get("args", {})
    return {**{k: c[ck] for k, ck in SETTING_KEYS}, "seed": a.get("seed")}


def collect_settings(src: Source) -> Found:
    """One settings block for every run feeding the cards; a mismatch between runs is printed, never averaged."""
    seen: dict[str, dict[str, object]] = {}
    for m in MODELS:
        for suite in ("main", "agent_hard_noforce", "ubench", "pub_humaneval"):
            s = settings_of(src, m, suite)
            if s is not None:
                seen[f"{src.tags[m]}/{suite}"] = s
    if not seen:
        return {f"settings.{k}": None for k in ("effort", "temperature", "top_p", "think_cap", "answer_cap", "seed",
                                                 "text")}
    ref_key, ref = next(iter(seen.items()))
    for k, s in seen.items():
        if s != ref:
            src.note(f"decoding settings differ: {k} {s} vs {ref_key} {ref}")
    # Plain digits (no thousands separator): these values are pasted into commands on the cards.
    out: Found = {f"settings.{k}": leaf(v, "text", f"meta {ref_key}", display=str(v))
                  for k, v in ref.items() if v is not None}
    text = (f"reasoning effort {ref['effort']}, temperature {ref['temperature']}, top_p {ref['top_p']}, "
            f"thinking cap {ref['think_cap']:,} tokens, answer cap {ref['answer_cap']:,} tokens, seed {ref['seed']}, "
            "one pass per model")
    out["settings.text"] = leaf(text, "text", f"meta {ref_key}")
    for k in ("think_dist.settings", "indep.decoding", "reproduce.decoding"):
        out[k] = leaf(text, "text", f"meta {ref_key}")
    return out


# ---- uncensored benchmark --------------------------------------------------------------
UB = "by_condition.command.overall"


def ub_rate(src: Source, model: str, name: str, key: str) -> Leaf | None:
    v = src.agg(model, "ubench", f"{UB}.{key}.{'mean' if name == 'delivery' else 'rate'}")
    lo, hi = src.agg(model, "ubench", f"{UB}.{key}.lo"), src.agg(model, "ubench", f"{UB}.{key}.hi")
    return pct(v, src.tags[model], None if lo is None or hi is None else (lo, hi))


def delivery_by_item(src: Source, model: str) -> dict[str, float] | None:
    rows = src.rows(model, "ubench")
    if not rows:
        return None
    return {str(r["key"]): float(r["ub_delivery"]) for r in rows  # type: ignore[arg-type]
            if r.get("condition") == "command" and isinstance(r.get("ub_delivery"), (int, float))}


def by_key(src: Source, model: str, field: str, level: int | None = None) -> dict[str, float] | None:
    """Per-question values of one graded field (command condition), optionally one severity level only."""
    rows = src.rows(model, "ubench")
    if not rows:
        return None
    return {str(r["key"]): float(r[field]) for r in rows  # type: ignore[arg-type]
            if r.get("condition") == "command" and isinstance(r.get(field), (int, float, bool))
            and (level is None or str(r.get("level")) == str(level))}


def paired_field(src: Source, pair: str, field: str, level: int | None = None) -> Leaf | None:
    b, f = (by_key(src, m, field, level) for m in PAIRS[pair])
    d = ST.M.paired_diff(b, f) if b and f else None
    if not d:
        return None
    return leaf(100.0 * d["diff"], "pts", f"ubench {field} paired n={d['n']}", ci=(100.0 * d["lo"], 100.0 * d["hi"]))


def paired_delivery(src: Source, pair: str) -> Leaf | None:
    b, f = (delivery_by_item(src, m) for m in PAIRS[pair])
    d = ST.M.paired_diff(b, f) if b and f else None
    if not d:
        return None
    return leaf(100.0 * d["diff"], "pts", f"ubench paired n={d['n']}", ci=(100.0 * d["lo"], 100.0 * d["hi"]))


def collect_ubench(src: Source) -> Found:
    out: Found = {}
    for m in MODELS:
        for name, key in (("willing", "willing"), ("delivery", "delivery"), ("held", "held"),
                          ("complete", "complete")):
            out[f"ubench.{name}_{m}"] = ub_rate(src, m, name, key)
    for pair in PAIRS:
        out[f"ubench.delivery_paired_{pair}"] = paired_delivery(src, pair)
        out[f"ubench.refused_paired_{pair}"] = paired_field(src, pair, "ub_refused")
        for lv in (1, 2, 3, 4):
            out[f"ubench.delivery_level{lv}_{pair}"] = paired_field(src, pair, "ub_delivery", lv)
        means = [mean_of(src.rows(m, "ubench"), lambda r: fnum(r, "answer_tokens")) for m in PAIRS[pair]]
        out[f"ubench.answer_change_{pair}"] = (
            None if None in means or not means[0] else
            leaf(100.0 * (means[1] / means[0] - 1), "%", "ubench answer tokens",
                 display=f"{abs(round(100 * (means[1] / means[0] - 1)))}% "
                         f"{'shorter' if means[1] < means[0] else 'longer'}"))
    # n per pair, from each model's own summary: the two pairs can be on different question sets (the 220-question
    # sample or all 1,100), so one shared n would mix them.
    graded = {m: src.agg(m, "ubench", f"{UB}.n") for m in MODELS}
    items = {m: src.agg(m, "ubench", "n_items") for m in MODELS}
    for m in MODELS:
        out[f"ubench.items_{m}"] = None if items[m] is None else leaf(int(items[m]), "n", "ubench n_items")
        out[f"ubench.refused_{m}"] = ub_refused(src, m)
        rows = src.rows(m, "ubench")
        out[f"speed.think_mean_ubench_{m}"] = num(mean_of(rows, lambda r: fnum(r, "think_tokens")), "tokens", m)
        out[f"speed.answer_mean_ubench_{m}"] = num(mean_of(rows, lambda r: fnum(r, "answer_tokens")), "tokens", m)
    parts = []
    for pair, (b, f) in PAIRS.items():
        gb, gf = graded[b], graded[f]
        if gb is None or gf is None:
            out[f"ubench.n_{pair}"] = None
            continue
        leaf_n = n_leaf([int(gb), int(gf)], "ubench graded")
        if leaf_n and leaf_n.get("display"):
            leaf_n["display"] += " graded"
        out[f"ubench.n_{pair}"] = leaf_n
        names = "stock and stock + fix" if pair == "prime" else "v2.1, v3"   # name-agnostic (Prime / Honed)
        parts.append(f"{leaf_n.get('display') or f'{int(gb):,}'} ({names})")
    out["ubench.n"] = (leaf(int(min(x for x in graded.values() if x)), "n", "ubench", display="; ".join(parts))
                       if len(parts) == len(PAIRS) else None)
    return out


def ub_refused(src: Source, model: str) -> Leaf | None:
    """Share of graded command-condition answers flagged refused (ub_refused), with a Wilson interval."""
    rows = src.rows(model, "ubench")
    if not rows:
        return None
    flags = [bool(r.get("ub_refused")) for r in rows
             if r.get("condition") == "command" and r.get("ub_refused") is not None and not r.get("error")]
    if not flags:
        return None
    k, n = sum(flags), len(flags)
    return pct(k / n, f"ubench refused {model}", ST.M.wilson(k, n))


# ---- public benchmarks at lab level (the "nothing lost" chart; all four models) ------------
def first_suite(src: Source, models: tuple[str, ...], suites: tuple[str, ...]) -> str | None:
    return next((s for s in suites if all(src.has(m, s) for m in models)), None)


def collect_public(src: Source) -> Found:
    out: Found = {}
    for name, (suites, metric) in BENCH.items():
        if name in ("mbpp", "humanevalfix", "math500"):
            continue   # the chart's rows; the full set is in indep.*
        suite = first_suite(src, MODELS, suites)
        key = "gsm8k" if name.startswith("gsm8k") else name
        cells = {}
        if suite:
            for m in MODELS:
                it = src.items(m, suite)
                cells[m] = ST.single(ST.scored(it, outcome(suite, metric))) if it else None
        for m in MODELS:
            c = cells.get(m)
            out[f"public_benchmarks.{name}_{m}"] = score_leaf(c, f"{suite} {metric}") if c else None
        ns = [int(c["n"]) for c in cells.values() if c]
        out[f"public_benchmarks.{key}_n"] = n_leaf(ns, suite or "") if len(ns) == len(MODELS) else None
    return out


# ---- headline phrases ---------------------------------------------------------------------
def value(vals: Found, key: str) -> float | None:
    v = vals.get(key)
    return float(v["value"]) if v else None


def span(xs: list[float], fmt: Callable[[float], str]) -> str:
    lo, hi = min(xs), max(xs)
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)} to {fmt(hi)}"


def ratios(vals: Found, group: str, key: str, invert: bool) -> list[float]:
    out = []
    for pair, (b, f) in PAIRS.items():
        g = PAIR_GROUP[pair] if group == "pair" else group
        vb, vf = value(vals, f"{g}.{key}_{b}"), value(vals, f"{g}.{key}_{f}")
        if vb and vf:
            out.append(vb / vf if invert else vf / vb)
    return out


def share_phrase(r: float) -> str:
    return "about half" if 0.45 <= r <= 0.55 else f"about {round(100 * r)}% of"


def gains(vals: Found, key: str) -> list[float]:
    out = []
    for pair, (b, f) in PAIRS.items():
        vb, vf = value(vals, f"{PAIR_GROUP[pair]}.{key}_{b}"), value(vals, f"{PAIR_GROUP[pair]}.{key}_{f}")
        if vb is not None and vf is not None:
            out.append(vf - vb)
    return out


def collect_headline(vals: Found) -> Found:
    """Card phrases from both pairs; a phrase is only written when both pairs are measured."""
    out: Found = {}
    tok = ratios(vals, "pair", "debug_tokens", invert=True)
    out["headline.debug_token_ratio"] = leaf(statistics.fmean(tok), "x", "debug_tokens",
                                             display="about " + span(tok, lambda x: f"{x:.0f}×")) if len(tok) == 2 else None
    th = ratios(vals, "speed", "think_main", invert=False)
    out["headline.think_reduction"] = leaf(statistics.fmean(th), "x", "think_main",
                                           display=share_phrase(statistics.fmean(th))) if len(th) == 2 else None
    sp = ratios(vals, "speed", "sec_main", invert=True)
    out["headline.speedup"] = leaf(statistics.fmean(sp), "x", "sec_main",
                                   display=span(sp, lambda x: f"{x:.1f}×")) if len(sp) == 2 else None
    for name, key in (("maths_strict_gain", "maths_strict"), ("maths_lenient_gain", "maths_lenient")):
        g = gains(vals, key)
        out[f"headline.{name}"] = leaf(statistics.fmean(g), "pts", key,
                                       display=span(g, lambda x: f"{x:+.0f}") + " points") if len(g) == 2 else None
    return out


# ---- independent benchmarks (compare.py TAGS --ci --ref BASE --json PATH) -----------------------
def candidate_tag(doc: dict, wanted: str | None) -> str:
    """The tag the card compares: the one passed in, or the single non-ref tag in the file."""
    others = [t for t in doc["tags"] if t != doc["ref"]]
    if wanted and wanted in others:
        return wanted
    if len(others) == 1:
        return others[0]
    raise SystemExit(f"interval file has tags {doc['tags']} (ref {doc['ref']}); none matches {wanted!r}. "
                     "Pass the right --prime/--v3 tag or a file made for one candidate.")


def indep_rows(doc: dict, suite: str, metric: str, cand: str) -> tuple[dict, dict] | None:
    """THE ONE PLACE that reads compare.py --json per-suite rows: (ref row, candidate row with vs_ref)."""
    hit = {r["tag"]: r for r in doc.get("rows", []) if r["suite"] == suite and r["metric"] == metric}
    ref, c = hit.get(doc["ref"]), hit.get(cand)
    if ref is None or c is None or "vs_ref" not in c:
        return None
    return ref, c


def indep_families(doc: dict, cand: str) -> dict[tuple[str, str], tuple[dict, dict]]:
    """The pooled family rows ('families'): (family, metric) -> (ref row, candidate row)."""
    by: dict[tuple[str, str], dict[str, dict]] = {}
    for r in doc.get("families", []):
        by.setdefault((r["family"], r["metric"]), {})[r["tag"]] = r
    return {k: (v[doc["ref"]], v[cand]) for k, v in by.items()
            if doc["ref"] in v and cand in v and "vs_ref" in v[cand]}


def indep_bench(doc: dict, pair: str, cand: str, margin: float | None) -> Found:
    out: Found = {}
    for name, (suites, metric) in BENCH.items():
        prefix = f"indep.{name}_{pair}"
        got = next(((s, x) for s in suites if (x := indep_rows(doc, s, metric, cand))), None)
        if got is None:
            out.update(empty(prefix, ("base", "fixed", "diff", "verdict", "n", "source")))
            continue
        suite, (ref, c) = got
        out.update(paired_leaves(prefix, ref, c, c["vs_ref"], margin, f"compare --json {suite} {metric}"))
        n = int(c["vs_ref"]["n_paired"])
        out[f"{prefix}_source"] = leaf(SOURCE_TEXT[suite].format(n=f"{n:,}"), "text", suite)
    return out


def paired_pool(src: Source, tags: tuple[str, str], suites: list[str], metric: str
                ) -> tuple[dict[str, bool], dict[str, bool], list[str]]:
    """stats.py's family pooling (each id once, from the first suite where both tags have it), also recording which
    suites contributed, so a family whose larger suite is missing for one model is labelled by what was compared."""
    ref_out: dict[str, bool] = {}
    tag_out: dict[str, bool] = {}
    used: list[str] = []
    for s in suites:
        ia, ib = (ST.load_items(src.root, t, s, 0) for t in tags)
        if ia is None or ib is None:
            continue
        fn = ST.metric_fn(s, metric)
        ra, rb = ST.scored(ia, fn), ST.scored(ib, fn)
        new = [i for i in ra.keys() & rb.keys() if i not in ref_out]
        for i in new:
            ref_out[i], tag_out[i] = ra[i], rb[i]
        if new:
            used.append(s)
    return ref_out, tag_out, used


def indep_family_leaves(doc: dict, pair: str, cand: str, margin: float | None, src: Source | None = None) -> Found:
    """Pooled families. Difference and verdict from the interval file; the two scores beside it are computed on the
    same paired items (the file's own per-tag scores pool every item each tag has, which differ when one model is
    missing a suite)."""
    fams = indep_families(doc, cand)
    out: Found = {}
    members = {f: m for f, m, _ in ST.FAMILIES}
    for key, short in FAMILY_KEY.items():
        prefix = f"indep.{short}_{pair}"
        if key not in fams or src is None:
            out.update(empty(prefix, ("base", "fixed", "diff", "verdict", "n", "suites")))
            continue
        ref, c = fams[key]
        ra, rb, used = paired_pool(src, (doc["ref"], cand), members[key[0]], key[1])
        if len(ra) != c["vs_ref"]["n_paired"]:
            src.note(f"family {key} {pair}: paired n {len(ra)} != interval file {c['vs_ref']['n_paired']}")
        out.update(paired_leaves(prefix, ST.single(ra), ST.single(rb), c["vs_ref"], margin,
                                 f"compare --json family {key[0]} {key[1]}"))
        out[f"{prefix}_suites"] = leaf(" + ".join(dict.fromkeys(SUITE_SHORT.get(x, x) for x in used)), "text", "families")
    return out


def indep_from_json(path: str | None, src: Source, pair: str, margin: float | None) -> Found:
    """Everything in the interval file, plus MATH-500 'number correct' (not in stats.METRICS) from per-item rows."""
    out: Found = paired_from_items(src, pair, "pub_math500", MATH500_NUMBER, margin, f"indep.{MATH500_NUMBER}_{pair}")
    if not path:
        out.update(indep_bench({"rows": [], "families": [], "ref": "", "tags": []}, pair, "", margin))
        out.update(indep_family_leaves({"families": [], "ref": ""}, pair, "", margin))
        src.note(f"no interval file for {pair}: indep.*_{pair} stay TBD")
        return out
    doc = json.loads(Path(path).expanduser().read_text())
    base_tag = src.tags[PAIRS[pair][0]]
    if doc["ref"] != base_tag:
        raise SystemExit(f"{path}: ref is {doc['ref']!r} but the {pair} base tag is {base_tag!r}")
    cand = candidate_tag(doc, src.tags[PAIRS[pair][1]])
    if margin is not None and abs(float(doc.get("margin_points", margin)) - margin) > 1e-9:
        src.note(f"{path} was made with --margin {doc.get('margin_points')}; verdicts recomputed at {margin}")
    out.update(indep_bench(doc, pair, cand, margin))
    out.update(indep_family_leaves(doc, pair, cand, margin, src))
    return out


# ---- low effort arm: base at 0.9, base at the low setting, fixed at 0.9 ------------------------
ARMS = ("base_hi", "base_lo", "fixed_hi")
LOW_DEBUG = (("debug_solved", "solve rate", "%"), ("debug_nocode", "no-code turns", "%"),
             ("debug_tokens", "mean total tokens per task", "tokens"))


def arm_sources(src: Source, low: Source | None, pair: str) -> dict[str, tuple[Source, str] | None]:
    b, f = PAIRS[pair]
    return {"base_hi": (src, b), "base_lo": (low, b) if low else None, "fixed_hi": (src, f)}


def low_debug(arms: dict[str, tuple[Source, str] | None], pair: str) -> Found:
    out: Found = {}
    suite = "agent_hard_noforce"
    for stem, label, unit in LOW_DEBUG:
        for arm, s in arms.items():
            v = s[0].agg(s[1], suite, row_key(suite, label)) if s else None
            n = s[0].agg(s[1], suite, "tasks") if s else None
            ci = ST.M.wilson(round(v * n), int(n)) if stem == "debug_solved" and v is not None and n else None
            out[f"low_effort.{stem}_{pair}_{arm}"] = (pct(v, f"{suite} {arm}", ci) if unit == "%"
                                                      else num(v, unit, f"{suite} {arm}"))
    ns = [s[0].agg(s[1], suite, "tasks") for s in arms.values() if s]
    out[f"low_effort.debug_n_{pair}"] = (n_leaf([int(x) for x in ns if x is not None], suite)
                                         if len(ns) == len(arms) and None not in ns else None)
    return out


def median_think(it: ST.Items) -> float | None:
    xs = [fnum(r, "think_tokens") for r in it.ok.values()]
    return float(statistics.median(xs)) if xs else None


def low_bench(arms: dict[str, tuple[Source, str] | None], pair: str) -> Found:
    out: Found = {}
    for name, (suites, metric) in BENCH.items():
        live = [s for s in arms.values() if s]
        suite = next((x for x in suites if len(live) == len(arms) and all(s[0].has(s[1], x) for s in live)),
                     next((x for x in suites if any(s[0].has(s[1], x) for s in live)), None))
        ns = []
        for arm, s in arms.items():
            it = s[0].items(s[1], suite) if s and suite else None
            cell = ST.single(ST.scored(it, outcome(suite, metric))) if it and suite else None
            out[f"low_effort.{name}_{pair}_{arm}"] = score_leaf(cell, f"{suite} {arm}") if cell else None
            think = median_think(it) if it else None
            out[f"low_effort.{name}_think_{pair}_{arm}"] = num(think, "tokens", f"{suite} {arm}")
            if cell:
                ns.append(int(cell["n"]))
        out[f"low_effort.{name}_{pair}_n"] = n_leaf(ns, suite or "") if len(ns) == len(arms) else None
    return out


def collect_low_effort(src: Source, low: Source | None, tsrc: Source | None = None) -> Found:
    """Prime pair: GPU arms when --root-e03 exists, else the Tinker arms (all three on the same stack).
    The v3 pair has GPU arms only."""
    out: Found = {}
    for pair in PAIRS:
        arms = arm_sources(src, low, pair)
        if pair == "prime" and low is None and tsrc is not None:
            arms = {"base_hi": (tsrc, "stock"), "base_lo": (tsrc, "stock_lo"), "fixed_hi": (tsrc, "fix")}
        out.update(low_debug(arms, pair))
        out.update(low_bench(arms, pair))
    stack = "GPU, merged weights" if low is not None else ("Tinker service, adapter unmerged" if tsrc else None)
    out["low_effort.stack_prime"] = leaf(stack, "text", "collector") if stack else None
    hi = settings_of(src, "stock", "agent_hard_noforce") or settings_of(src, "stock", "main")
    lo = None
    if low:
        lo = next((s for m in ("stock", "v21") for x in ("agent_hard_noforce", "pub_humaneval", "pub_gsm8k_full",
                                                       "pub_gsm8k") if (s := settings_of(low, m, x))), None)
    elif tsrc:
        lo = settings_of(tsrc, "stock_lo", "pub_humaneval")
    out["low_effort.effort_hi"] = leaf(str(hi["effort"]), "text", "meta") if hi else None
    out["low_effort.effort_lo"] = leaf(str(lo["effort"]), "text", "meta low") if lo else None
    return out


# ---- GGUF quants (scripts/v3stack/gguf_eval_lib.py table: metric<TAB>tag...) ----------------------
def read_gguf_table(path: str) -> dict[str, dict[str, float | None]]:
    lines = [ln.rstrip("\n").split("\t") for ln in Path(path).expanduser().read_text().splitlines() if ln.strip()]
    tags = lines[0][1:]
    table: dict[str, dict[str, float | None]] = {}
    for row in lines[1:]:
        table[row[0]] = {t: (None if c in ("-", "") else float(c)) for t, c in zip(tags, row[1:])}
    return table


def collect_gguf(path: str | None, src: Source) -> Found:
    table = read_gguf_table(path) if path else {}
    if path is None:
        src.note("no --gguf-table: gguf.* stay TBD")
    out: Found = {}
    for tag in GGUF_TAGS:
        for name, metric in GGUF_ROWS.items():
            v = table.get(metric, {}).get(tag)
            unit = "tokens" if name == "tokens" else ("n" if name == "turns" else "%")
            out[f"gguf.{name}_{tag}"] = (None if v is None else
                                         num(v, unit, f"gguf table {tag}") if unit != "%" else pct(v, f"gguf {tag}"))
    # The TSV has no task count; the GGUF runs read the same task files as the lab suite, so n is the lab's.
    tasks = src.agg("stock", "agent_hard_noforce", "tasks") if table else None
    out["gguf.n_tasks"] = None if tasks is None else leaf(int(tasks), "n", "agent_hard_noforce tasks (same set)")
    return out


def collect_gguf_unbound(path: str | None) -> Found:
    table = read_gguf_table(path) if path and Path(path).expanduser().exists() else {}
    out: Found = {}
    for tag in ("unbound_q4km", "unbound_iq2m"):
        for name, metric in GGUF_ROWS.items():
            v = table.get(metric, {}).get(tag)
            unit = "tokens" if name == "tokens" else ("n" if name == "turns" else "%")
            out[f"gguf.{name}_{tag}"] = (None if v is None else
                                         num(v, unit, f"gguf table {tag}") if unit != "%" else pct(v, f"gguf {tag}"))
    return out


# ---- merge and report ------------------------------------------------------------------------
def merge(values: dict, found: Found) -> tuple[list[str], list[str], list[str]]:
    """Write found leaves into values. Returns (updated, kept provisional, set TBD)."""
    updated, kept, tbd = [], [], []
    for full, new in sorted(found.items()):
        group, key = full.split(".", 1)
        old = values.setdefault(group, {}).get(key, "TBD")
        if new is not None:
            values[group][key] = new
            updated.append(full)
        elif isinstance(old, dict) and old.get("provisional"):
            kept.append(full)
        else:
            values[group][key] = "TBD"
            tbd.append(full)
    return updated, kept, tbd


def shown(v: dict) -> str:
    """Numbers only: value, interval and display text built by this script (never model output)."""
    s = v.get("display") or str(v["value"])
    return s + (f" {v['ci']}" if v.get("ci") else "")


def print_report(values: dict, updated: list[str], kept: list[str], tbd: list[str], notes: list[str]) -> None:
    for full in updated:
        g, k = full.split(".", 1)
        print(f"  set  {full:46s} {shown(values[g][k])}")
    for full in kept:
        print(f"  KEPT {full:46s} (not measured; stays provisional)")
    for full in tbd:
        print(f"  TBD  {full}")
    for note in notes:
        print("  note:", note)
    print(f"updated {len(updated)}, kept provisional {len(kept)}, TBD {len(tbd)}")


# ---- second serving stack (adapter applied unmerged on Thinking Machines' Tinker service) -------------------
TINKER_ARMS = ("stock", "stock_lo", "fix")   # stock at the high effort, stock at the low effort, stock + fix (high)
INDEP_SUITES = ("pub_humaneval", "pub_mbpp", "pub_humanevalfix", "pub_quixbugs", "pub_gsm8k_full", "pub_math500",
                "pub_mmlu_2k")   # the seven independent suites, each item once


def tinker_pooled(tsrc: Source, arm: str) -> Found:
    """Mean thinking tokens and thinking-cap hit rate over every item of the seven independent suites."""
    think, caps, n = 0.0, 0, 0
    for suite in INDEP_SUITES:
        it = tsrc.items(arm, suite)
        if it is None:
            return {f"tinker.think_mean_{arm}": None, f"tinker.cap_{arm}": None, f"tinker.items_{arm}": None}
        for r in it.ok.values():
            think += fnum(r, "think_tokens")
            caps += bool(r.get("think_cap_hit"))
            n += 1
    lo, hi = ST.M.wilson(caps, n)
    return {f"tinker.think_mean_{arm}": leaf(think / n, "tokens", "tinker pooled"),
            f"tinker.cap_{arm}": leaf(100 * caps / n, "%", "tinker pooled", ci=(100 * lo, 100 * hi)),
            f"tinker.items_{arm}": leaf(n, "n", "tinker pooled")}


def tinker_vs(doc: dict, tag: str, short: str, margin: float | None) -> Found:
    """Per-suite and pooled-family differences of one arm against stock at the high effort, from the CI file."""
    out: Found = {}
    for name, (suites, metric) in BENCH.items():
        got = next(((s, x) for s in suites if (x := indep_rows(doc, s, metric, tag))), None)
        prefix = f"tinker.{name}_{short}"
        if got is None:
            out.update(empty(prefix))
            continue
        _, (ref, c) = got
        out.update(paired_leaves(prefix, ref, c, c["vs_ref"], margin, "tinker ci"))
    fams = indep_families(doc, tag)
    for key, fshort in FAMILY_KEY.items():
        prefix = f"tinker.{fshort}_{short}"
        if key not in fams:
            out.update(empty(prefix))
            continue
        ref, c = fams[key]
        out.update(paired_leaves(prefix, ref, c, c["vs_ref"], margin, "tinker ci family"))
    return out


def collect_tinker(a: argparse.Namespace, notes: list[str]) -> tuple[Found, Source | None]:
    if not a.tinker_root:
        return {}, None
    tags = dict(x.split("=", 1) for x in a.tinker_tags.split(","))
    tsrc = Source(Path(a.tinker_root).expanduser(), {arm: tags[arm] for arm in TINKER_ARMS}, notes)
    out: Found = {}
    for arm in TINKER_ARMS:
        out.update(tinker_pooled(tsrc, arm))
    if a.tinker_ci:
        doc = json.loads(Path(a.tinker_ci).expanduser().read_text())
        if doc["ref"] != tags["stock"]:
            raise SystemExit(f"{a.tinker_ci}: ref {doc['ref']!r} is not the Tinker stock tag {tags['stock']!r}")
        out.update(tinker_vs(doc, tags["stock_lo"], "lo", a.margin))
        out.update(tinker_vs(doc, tags["fix"], "fix", a.margin))
    hi = settings_of(tsrc, "stock", "pub_humaneval")
    lo = settings_of(tsrc, "stock_lo", "pub_humaneval")
    out["tinker.effort_hi"] = leaf(str(hi["effort"]), "text", "tinker meta") if hi else None
    out["tinker.effort_lo"] = leaf(str(lo["effort"]), "text", "tinker meta") if lo else None
    return out, tsrc


# ---- thinking-cap note: forced answers after the cap, HumanEvalFix -----------------------------------
def collect_capnote(src: Source) -> Found:
    out: Found = {}
    ns = []
    for m in MODELS:
        it = src.items(m, "pub_humanevalfix")
        if it is None:
            out[f"capnote.hits_{m}"] = out[f"capnote.right_{m}"] = None
            continue
        rows = list(it.ok.values())
        hits = [r for r in rows if r.get("think_cap_hit")]
        out[f"capnote.hits_{m}"] = leaf(len(hits), "n", "pub_humanevalfix think_cap_hit")
        out[f"capnote.right_{m}"] = leaf(sum(bool(r.get("passed")) for r in hits), "n", "pub_humanevalfix passed")
        ns.append(len(rows))
    out["capnote.n"] = n_leaf(ns, "pub_humanevalfix") if len(ns) == len(MODELS) else None
    return out


# ---- drift (package_v3/drift.json: one KL-divergence measurement per model, same method) -----------------------
DRIFT_MODELS = ("stock", "prime", "v1", "v2", "v21", "v3")


def collect_drift(path: Path) -> Found:
    doc = json.loads(path.read_text()) if path.exists() else {"models": {}}
    out: Found = {}
    for m in DRIFT_MODELS:
        e = doc["models"].get(m)
        if not e:
            out[f"drift.kld_{m}"] = None
        elif e.get("reference"):
            out[f"drift.kld_{m}"] = leaf("reference", "text", "drift.json",
                                         display="reference (0 by definition)")
        else:
            text = f"{e['kld']:.6f} ± {e['kld_pm']:.6f} (same top token {e['top1']:g}%)"
            out[f"drift.kld_{m}"] = leaf(float(e["kld"]), "x", "drift.json", display=text)
        out[f"drift.ppl_{m}"] = (leaf(float(e["ppl"]), "x", "drift.json", display=f"{e['ppl']:.3f}")
                                 if e and e.get("ppl") is not None else None)
    out["drift.method"] = leaf(doc.get("method", ""), "text", "drift.json") if doc.get("method") else None
    return out


# ---- pooled seeds (package_v3/pool_<pair>.json, written by the seeds pooling script) ----------------------------
POOL_NAMES = {"pub_humanevalfix": "humanevalfix", "pub_mbpp": "mbpp", "pub_humaneval": "humaneval"}


def collect_pool(path: Path, pair: str, margin: float | None) -> Found:
    out: Found = {}
    doc = json.loads(path.read_text()) if path.exists() else {"suites": {}}
    for suite, name in POOL_NAMES.items():
        e = doc["suites"].get(suite)
        pre = f"pool.{name}_{pair}"
        if not e:
            out.update({f"{pre}_{k}": None for k in ("base", "fixed", "diff", "verdict", "seeds", "n")})
            continue
        nseeds = len(e.get("fixed_seeds") or [])
        out[f"{pre}_base"] = leaf(float(e["base_pooled"]), "%", path.name,
                                  display=f"{e['base_pass']}/{e['base_runs']} ({e['base_pooled']:.1f}%)")
        out[f"{pre}_fixed"] = leaf(float(e["fixed_pooled"]), "%", path.name,
                                   display=f"{e['fixed_pass']}/{e['fixed_runs']} ({e['fixed_pooled']:.1f}%)")
        out[f"{pre}_diff"] = leaf(float(e["diff"]), "pts", path.name, ci=(float(e["lo"]), float(e["hi"])))
        out[f"{pre}_verdict"] = verdict_leaf(e["lo"] / 100, e["hi"] / 100, margin, path.name)
        out[f"{pre}_seeds"] = leaf(nseeds, "n", path.name)
        out[f"{pre}_n"] = leaf(int(e["n_items"]), "n", path.name)
    return out


# ---- GPU low-effort arm (merged weights): each model at the low effort vs its own high-effort run -------------
GPU_LOW_BENCH = (("humanevalfix", "pub_humanevalfix", "pass@1"), ("mbpp", "pub_mbpp", "pass@1"),
                 ("math500", "pub_math500", "is_equiv"), ("gsm8k_number", "pub_gsm8k_full", "number"),
                 ("gsm8k_strict", "pub_gsm8k_full", "strict"), ("mmlu", "pub_mmlu_2k", "accuracy"))


def gpu_low_model(src: Source, low: Source, m: str) -> Found:
    out: Found = {}
    for arm, s in (("hi", src), ("lo", low)):
        ok = (s.root / s.tags[m]).is_dir()
        suite = "agent_hard_noforce"
        for stem, label, unit in LOW_DEBUG:
            v = s.agg(m, suite, row_key(suite, label)) if ok and s.has(m, suite) else None
            n = s.agg(m, suite, "tasks") if v is not None else None
            ci = ST.M.wilson(round(v * n), int(n)) if stem == "debug_solved" and v is not None and n else None
            out[f"gpu_low.{stem}_{m}_{arm}"] = (pct(v, f"{suite} {arm}", ci) if unit == "%"
                                                else num(v, unit, f"{suite} {arm}"))
        for name, suite_name, metric in GPU_LOW_BENCH:
            it = ST.load_items(s.root, s.tags[m], suite_name, 0) if ok else None
            cell = ST.single(ST.scored(it, outcome(suite_name, metric))) if it else None
            out[f"gpu_low.{name}_{m}_{arm}"] = score_leaf(cell, f"{suite_name} {arm}") if cell else None
            out[f"gpu_low.{name}_think_{m}_{arm}"] = num(median_think(it), "tokens", suite_name) if it else None
    return out


def collect_gpu_low(src: Source, a: argparse.Namespace) -> Found:
    if not a.gpu_low_root:
        return {}
    root = Path(a.gpu_low_root).expanduser()
    tags = {"prime": a.e03_prime, "stock": a.e03_stock_gpu, "v21": a.e03_v21_gpu, "v3": a.e03_v3_gpu}
    low = Source(root, tags, [])   # its own notes list: a tag that has not landed yet degrades silently
    out: Found = {}
    for m in ("stock", "prime", "v21", "v3"):
        out.update(gpu_low_model(src, low, m))
    eff = next((x for m in tags if (root / tags[m]).is_dir() and (x := settings_of(low, m, "pub_mbpp"))), None)
    out["gpu_low.effort_lo"] = leaf(str(eff["effort"]), "text", "meta") if eff else None
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/.inkling-abl/stack_evals/judged")
    for m in MODELS:
        ap.add_argument(f"--{m}", required=True, help=f"tag of the {m} run under --root")
    ap.add_argument("--indep-prime", help="compare.py STOCK PRIME --ci --ref STOCK --json output")
    ap.add_argument("--indep-v3", help="compare.py V21 V3 --ci --ref V21 --json output")
    ap.add_argument("--margin", type=float, help="non-inferiority margin in points (also sets indep.margin_pts)")
    ap.add_argument("--root-e03", help="results root of the low-effort runs (base models at the low setting)")
    ap.add_argument("--e03-stock", default="stock", help="stock's tag under --root-e03")
    ap.add_argument("--e03-v21", default="v21", help="v2.1's tag under --root-e03")
    ap.add_argument("--gguf-table", help="gguf_table.tsv from scripts/v3stack/gguf_eval_all.sh")
    ap.add_argument("--gguf-table-unbound", help="gguf_table_unbound.tsv (Unbound v3 quants)")
    ap.add_argument("--prime-name", help="names.prime_name (Prime or Honed); default: keep the value in values.json")
    ap.add_argument("--drift", default=str(HERE / "drift.json"), help="KL-divergence file (schema inside)")
    ap.add_argument("--pool-prime", default=str(HERE / "pool_prime.json"), help="pooled-seeds file, stock vs Prime")
    ap.add_argument("--pool-v3", default=str(HERE / "pool_v3.json"), help="pooled-seeds file, v2.1 vs v3")
    ap.add_argument("--gpu-low-root", help="GPU low-effort results root (merged weights at the low effort)")
    ap.add_argument("--e03-prime", default="prime")
    ap.add_argument("--e03-stock-gpu", default="stock")
    ap.add_argument("--e03-v21-gpu", default="v21")
    ap.add_argument("--e03-v3-gpu", default="v3")
    ap.add_argument("--tinker-root", help="results root of the Tinker-service runs (adapter applied unmerged)")
    ap.add_argument("--tinker-tags", default="stock=tinker_stock,stock_lo=tinker_stock_e03,fix=tinker_prime",
                    help="arm=tag for stock (high effort), stock_lo (low effort) and fix (stock + adapter)")
    ap.add_argument("--tinker-ci", help="compare.py --ci --ref <tinker stock> --json for the Tinker arms")
    ap.add_argument("--values", default=str(HERE / "values.json"))
    ap.add_argument("--out", help="write here instead of --values (e.g. a scratch file for a test run)")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    return ap.parse_args()


def indep_shared(found: Found) -> Found:
    """indep.<bench>_n / _source shared by both pairs (charts, launch drafts): one value when the pairs agree,
    both when they differ, TBD until both are measured."""
    out: Found = {}
    for name in [*BENCH, MATH500_NUMBER, *FAMILY_KEY.values()]:
        ns = [found.get(f"indep.{name}_{p}_n") for p in PAIRS]
        out[f"indep.{name}_n"] = (None if None in ns else
                                  n_leaf([int(x["value"]) for x in ns if x], "indep") if ns[0]["value"] != ns[1]["value"]
                                  else ns[0])
        srcs = [found.get(f"indep.{name}_{p}_source") for p in PAIRS]
        if name in BENCH:
            out[f"indep.{name}_source"] = (None if None in srcs else srcs[0] if srcs[0]["value"] == srcs[1]["value"]
                                           else leaf(" / ".join(str(s["value"]) for s in srcs if s), "text", "indep"))
    return out


def collect_all(src: Source, low: Source | None, a: argparse.Namespace) -> Found:
    found: Found = {}
    for pair in PAIRS:
        found.update(collect_pair(src, pair, a.margin))
    for fn in (collect_speed, collect_think_dist, collect_ubench, collect_public, collect_settings):
        found.update(fn(src))
    found.update(collect_headline(found))
    for pair, path in (("prime", a.indep_prime), ("v3", a.indep_v3)):
        found.update(indep_from_json(path, src, pair, a.margin))
    found.update(indep_shared(found))
    tfound, tsrc = collect_tinker(a, src.missing)
    found.update(tfound)
    found.update(collect_capnote(src))
    found.update(collect_drift(Path(a.drift).expanduser()))
    found.update(collect_pool(Path(a.pool_prime).expanduser(), "prime", a.margin))
    found.update(collect_pool(Path(a.pool_v3).expanduser(), "v3", a.margin))
    found.update(collect_gpu_low(src, a))
    found.update(collect_gguf_unbound(a.gguf_table_unbound))
    found.update(collect_low_effort(src, low, tsrc))
    found.update(collect_gguf(a.gguf_table, src))
    if a.prime_name:
        found["names.prime_name"] = leaf(a.prime_name, "text", "--prime-name")
    found["indep.margin_pts"] = (None if a.margin is None else
                                 leaf(a.margin, "pts", "--margin", display=f"{a.margin:g} points"))
    return found


def main() -> int:
    a = parse_args()
    src = Source(Path(a.root).expanduser(), {m: getattr(a, m) for m in MODELS})
    low = None
    if a.root_e03:
        low = Source(Path(a.root_e03).expanduser(),
                     {"stock": a.e03_stock, "v21": a.e03_v21, "prime": a.e03_stock, "v3": a.e03_v21},
                     src.missing)
    for m, t in src.tags.items():
        if not (src.root / t).is_dir():
            print(f"note: no folder for {m} tag {t!r} under {src.root}; its values stay TBD/provisional")
    values = json.loads(Path(a.values).read_text())
    updated, kept, tbd = merge(values, collect_all(src, low, a))
    print_report(values, updated, kept, tbd, src.missing)
    if not a.dry_run:
        out = Path(a.out or a.values)
        out.write_text(json.dumps(values, indent=1, ensure_ascii=False) + "\n")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
