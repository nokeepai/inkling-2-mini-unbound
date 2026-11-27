#!/usr/bin/env python3
"""Render the v3 release cards from templates/ + values.json into dist/<repo>/README.md.

  python3 package_v3/render.py            # draft (default): TBD shown as [TBD group.key], provisional marked †
  python3 package_v3/render.py --final    # refuses (exit 1) if any used value is TBD or provisional

Placeholders: {{group.key}} (a leaf of values.json), derived keys under names.* and auto.* (see derived()),
and {{#if flags.x}} ... {{else}} ... {{/if}} blocks (no nesting). A placeholder that names no value is always an
error, in both modes, so a typo can never ship as literal braces.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

HERE = Path(__file__).resolve().parent
HF = "https://huggingface.co"
V21_VALUES = HERE.parent / "package_v21" / "values.json"  # published v2.1 numbers, read-only, as v21card.<KEY>
# {{group.key}} or {{group.key:plain}}; ":plain" drops the bracketed interval and a trailing ".0" so headline
# bullets stay short ("100.0% [79.6, 100.0]" -> "100%"). The full value stays in the tables below.
PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+\.[A-Za-z0-9_]+)(:plain)?\}\}")
INTERVAL = re.compile(r"\s*\[[^\]]*\]")


def plain(s: str) -> str:
    return re.sub(r"(\d)\.0(%| s\b|$)", r"\1\2", INTERVAL.sub("", s)).strip()
IF_BLOCK = re.compile(r"\{\{#if ([a-z0-9_]+\.[a-z0-9_]+)\}\}(.*?)(?:\{\{else\}\}(.*?))?\{\{/if\}\}", re.S)
Scalar = Union[str, int, float, bool]


@dataclass(frozen=True)
class Leaf:
    key: str
    value: Scalar | None  # None = TBD
    unit: str = "text"
    provisional: bool = False
    display: str | None = None
    ci: tuple[float, float] | None = None  # 95% interval, same unit as value


# ---- loading ------------------------------------------------------------------
def load_values(path: Path) -> dict[str, Leaf]:
    raw = json.loads(path.read_text())
    out: dict[str, Leaf] = {}
    for group, items in raw.items():
        if group.startswith("_"):
            continue
        for key, v in items.items():
            k = f"{group}.{key}"
            if v == "TBD":
                out[k] = Leaf(k, None)
            else:
                ci = tuple(v["ci"]) if v.get("ci") else None
                out[k] = Leaf(k, v["value"], v.get("unit", "text"), bool(v.get("provisional")), v.get("display"), ci)
    return out


def load_v21_card() -> dict[str, Leaf]:
    """v2.1's published card values: history rows and the cross-release comparison, which v3 did not re-run."""
    raw: dict[str, str] = json.loads(V21_VALUES.read_text())
    return {f"v21card.{k}": Leaf(f"v21card.{k}", v) for k, v in raw.items()}


def fmt(leaf: Leaf) -> str:
    """Card text for a filled leaf; one place decides number style so every card agrees."""
    if leaf.display:
        return leaf.display
    v, u = leaf.value, leaf.unit
    if isinstance(v, bool) or u in ("text", "bool"):
        return str(v)
    assert isinstance(v, (int, float))
    if u == "pts" and abs(v) < 0.05:
        v = 0.0  # no "-0.0 pts"
    text = {"%": f"{v:.1f}%", "pts": f"{v:+.1f} pts", "x": f"{v:.1f}×", "tokens": f"{v:,.0f}",
            "s": f"{v:.1f} s", "n": f"{int(v):,}"}.get(u, f"{v}")
    if leaf.ci:
        lo, hi = leaf.ci
        text += f" [{lo:+.1f}, {hi:+.1f}]" if u == "pts" else f" [{lo:.1f}, {hi:.1f}]"
    return text


# ---- derived values -------------------------------------------------------------
# (label, group, base key, fixed key, n key, paired-difference key or "", higher is better): the lab checks for the
# "what got worse" lists. Public benchmarks come from INDEP below (full runs, paired intervals), not from here.
CHECKS: list[tuple[str, str, str, str, str, str, bool]] = [
    ("Hard debugging solved", "{pair}", "debug_solved_{b}", "debug_solved_{f}", "{pair}.debug_n_tasks", "", True),
    ("Debugging turns with no code written", "{pair}", "debug_nocode_{b}", "debug_nocode_{f}", "", "", False),
    ("Answer quality, independent grader", "{pair}", "quality_good_{b}", "quality_good_{f}", "{pair}.quality_n", "",
     True),
    ("Creative writing quality", "{pair}", "creative_good_{b}", "creative_good_{f}", "{pair}.creative_n", "", True),
    ("Roleplay quality", "{pair}", "rp_good_{b}", "rp_good_{f}", "{pair}.rp_n", "", True),
    ("Code problems passed first try (our own set)", "{pair}", "code_pass_{b}", "code_pass_{f}", "{pair}.code_n", "{pair}.code_pass_diff",
     True),
    ("Hard maths, strict", "{pair}", "maths_strict_{b}", "maths_strict_{f}", "{pair}.maths_n",
     "{pair}.maths_strict_diff", True),
    ("Hard maths, number correct", "{pair}", "maths_lenient_{b}", "maths_lenient_{f}", "{pair}.maths_n",
     "{pair}.maths_lenient_diff", True),
    ("uncensoredbench delivery score", "ubench", "delivery_{b}", "delivery_{f}", "ubench.n_{p}",
     "ubench.delivery_paired_{p}", True),
    ("Thinking-cap hits, uncensoredbench", "speed", "cap_ubench_{b}", "cap_ubench_{f}", "ubench.n_{p}", "", False),
    ("uncensoredbench willing", "ubench", "willing_{b}", "willing_{f}", "ubench.n_{p}", "", True),
    ("uncensoredbench held back", "ubench", "held_{b}", "held_{f}", "ubench.n_{p}", "", False),
    ("uncensoredbench complete", "ubench", "complete_{b}", "complete_{f}", "ubench.n_{p}", "", True),
]
# Willingness rows are a regression only for Unbound; for Prime they are a neutral "did the fix change what it
# answers" check, shown in its own table.
UNBOUND_ONLY = {"uncensoredbench willing", "uncensoredbench held back", "uncensoredbench complete"}
PAIRS = {"prime": ("prime_vs_stock", "stock", "prime"), "v3": ("v3_vs_v21", "v21", "v3")}
CheckRow = tuple[str, Leaf, Leaf, Union[Leaf, None], Union[Leaf, None], bool]


def check_rows(vals: dict[str, Leaf], pair: str) -> list[CheckRow]:
    group_name, b, f = PAIRS[pair]
    rows = []
    for label, group, bk, fk, nk, dk, hib in CHECKS:
        if pair == "prime" and label in UNBOUND_ONLY:
            continue
        g = group.format(pair=group_name)
        n = vals.get(nk.format(pair=group_name, p=pair)) if nk else None
        d = vals.get(dk.format(pair=group_name, p=pair)) if dk else None
        rows.append((label, vals[f"{g}.{bk.format(b=b)}"], vals[f"{g}.{fk.format(f=f)}"], n, d, hib))
    return rows


INDEP = [("HumanEval", "humaneval"), ("MBPP sanitized", "mbpp"), ("HumanEvalFix", "humanevalfix"),
         ("QuixBugs", "quixbugs"), ("GSM8K, strict", "gsm8k_strict"), ("GSM8K, number correct", "gsm8k_lenient"),
         ("MATH-500, strict format", "math500"), ("MATH-500, number correct", "math500_number"), ("MMLU", "mmlu")]


def is_worse(base: Leaf, fixed: Leaf, higher_better: bool) -> bool:
    """Lower at the precision the card prints (one decimal), so a listed dip is one the reader can see."""
    assert isinstance(base.value, (int, float)) and isinstance(fixed.value, (int, float))
    b, f = round(base.value, 1), round(fixed.value, 1)
    return f < b if higher_better else f > b


def noise_word(diff: Leaf | None) -> str:
    """'beyond noise' only when a paired 95% interval excludes zero; lab checks have no interval, so say so."""
    if diff is None or diff.value is None or diff.ci is None:
        return "one pass, no paired interval"
    return "beyond noise" if diff.ci[1] < 0 else "within noise"


def bullet(label: str, base: Leaf, fixed: Leaf, n: Leaf | None, diff: Leaf | None) -> str:
    size = f", n = {fmt(n)}" if n and n.value is not None else ""
    change = f", paired difference {fmt(diff)}" if diff and diff.value is not None else ""
    prov = " (provisional)" if any(x is not None and x.provisional for x in (base, fixed, diff)) else ""
    return f"- **{label}:** {fmt(base)} → {fmt(fixed)}{size}{change}; {noise_word(diff)}{prov}."


def indep_rows(vals: dict[str, Leaf], pair: str) -> list[CheckRow]:
    return [(f"{label} (independent)", vals[f"indep.{k}_{pair}_base"], vals[f"indep.{k}_{pair}_fixed"],
             vals[f"indep.{k}_{pair}_n"], vals[f"indep.{k}_{pair}_diff"], True) for label, k in INDEP]


def below_zero(diff: Leaf | None) -> bool:
    return diff is not None and diff.value is not None and diff.ci is not None and diff.ci[1] < 0


def worse_list(vals: dict[str, Leaf], pair: str, base_name: str) -> tuple[str, list[str]]:
    """Checks below the base: first those whose paired interval lies entirely below zero (a measured loss), then
    every other dip, inside noise. 'Nothing below' only when neither list nor a hand-written note has an entry."""
    beyond, inside, pending, used = [], [], [], []
    for label, base, fixed, n, diff, hib in check_rows(vals, pair) + indep_rows(vals, pair):
        used += [x.key for x in (base, fixed, n, diff) if x is not None]
        if base.value is None or fixed.value is None:
            pending.append(label)
        elif below_zero(diff):
            beyond.append(bullet(label, base, fixed, n, diff))
        elif is_worse(base, fixed, hib):
            inside.append(bullet(label, base, fixed, n, diff))
    note = vals[f"regressions.{pair}_note"]
    used.append(note.key)
    if note.value:
        inside.append(f"- {note.value}" + (" (provisional)" if note.provisional else "") + ".")
    parts = []
    if beyond:
        parts.append("**Below " + base_name + " beyond noise (whole paired interval below zero):**\n\n" + "\n".join(beyond))
    if inside:
        parts.append("**Lower, but inside noise:**\n\n" + "\n".join(inside))
    if not parts:
        parts.append(f"- Nothing measured came out below {base_name}.")
    if pending:
        parts.append(f"- Not yet measured: {', '.join(pending)}.")
    return "\n\n".join(parts), used


def beyond_noise_labels(vals: dict[str, Leaf], pair: str) -> list[str]:
    """Exactly the rows whose paired interval lies entirely below zero (what the rename rule is about)."""
    return [label for label, base, fixed, n, diff, _ in check_rows(vals, pair) + indep_rows(vals, pair)
            if base.value is not None and fixed.value is not None and below_zero(diff)]


UB_PAIRS = {"prime": ("stock", "prime"), "v3": ("v21", "v3")}


def ubench_full(vals: dict[str, Leaf], pair: str) -> bool | None:
    """True when both models of the pair ran all of uncensoredbench, False for the sample, None if unknown."""
    total = vals["ubench.n_total"].value
    its = [vals.get(f"ubench.items_{m}") for m in UB_PAIRS[pair]]
    if total is None or any(x is None or x.value is None for x in its):
        return None
    return all(int(x.value) >= int(total) for x in its)


def ubench_scope(vals: dict[str, Leaf], pair: str) -> Leaf:
    """How much of uncensoredbench a pair ran, worded from the data (items per model), never assumed."""
    key = f"auto.ubench_scope_{pair}"
    full, total, n = ubench_full(vals, pair), vals["ubench.n_total"], vals[f"ubench.n_{pair}"]
    if full is None or n.value is None:
        return Leaf(key, None)
    if full:
        graded = fmt(n)
        text = f"all {int(total.value):,} questions ({graded}{'' if 'graded' in graded else ' graded'})"
    else:
        text = (f"{int(n.value):,} of its {int(total.value):,} questions (an even sample across categories and "
                "levels)")
    return Leaf(key, text)


def ubench_limit(vals: dict[str, Leaf], pair: str) -> Leaf:
    """The reproduce line's uncensoredbench scope for this pair."""
    full = ubench_full(vals, pair)
    if full is None:
        return Leaf(f"auto.ubench_limit_{pair}", None)
    text = ("all uncensoredbench questions (no --limit)" if full else
            "the uncensoredbench sample adds --limit 220 for the ubench suite only")
    return Leaf(f"auto.ubench_limit_{pair}", text)


# ---- alerts and the cross-version table ------------------------------------------------
INDEP_LABEL = dict((k, label) for label, k in INDEP)


POOLED = (("HumanEvalFix", "humanevalfix"), ("MBPP", "mbpp"), ("HumanEval", "humaneval"))


def pooled_line(vals: dict[str, Leaf], pair: str, base_name: str) -> tuple[list[str], set[str], list[str]]:
    """Pooled multi-seed results, which supersede the single-seed row for the first screen."""
    cells, covered, used = [], set(), []
    for label, k in POOLED:
        d, v, nseeds = (vals.get(f"pool.{k}_{pair}_{x}") for x in ("diff", "verdict", "seeds"))
        if d is None or d.value is None or v is None or v.value is None:
            continue
        used += [d.key, v.key, nseeds.key]
        covered.add(k)
        word = {"WORSE": "worse", "inconclusive": "inconclusive, a dip that the pooled seeds neither confirm nor "
                "rule out"}.get(str(v.value), f"no worse than {base_name}")
        cell = f"{label}, {fmt(nseeds)} seeds pooled: {fmt(d)}, {word}"
        tk = vals.get(f"tinker.{k}_fix_diff") if pair == "prime" else None
        if tk is not None and tk.value is not None and str(v.value) != "non-inferior":
            cell += f" (on Thinking Machines' hosted service the same benchmark gave {fmt(tk)})"
            used.append(tk.key)
        cells.append(cell)
    return cells, covered, used


def alert(vals: dict[str, Leaf], pair: str, base_name: str) -> tuple[Leaf, list[str]]:
    """First-screen lines: pooled multi-seed results where they exist, then every other public benchmark whose
    single-run verdict is WORSE or inconclusive. Empty when none."""
    worse, open_ = [], []
    pooled, covered, used = pooled_line(vals, pair, base_name)
    for label, k in INDEP:
        v, dleaf, n = (vals[f"indep.{k}_{pair}_{x}"] for x in ("verdict", "diff", "n"))
        used += [v.key, dleaf.key, n.key]
        if v.value is None or dleaf.value is None or k in covered:
            continue
        cell = f"{label} ({fmt(dleaf)}, n = {fmt(n)})"
        if v.value == "WORSE":
            worse.append(cell)
        elif v.value == "inconclusive":
            open_.append(cell)
    parts = []
    if pooled:
        parts.append("**Repeated over several seeds (paired, items resampled as clusters):** " + "; ".join(pooled) + ".")
    if worse:
        parts.append(f"**Worse than {base_name}, whole interval below zero:** " + "; ".join(worse) + ".")
    if open_:
        parts.append(f"**Single run, not shown to be within the margin (a loss is not ruled out):** "
                     + "; ".join(open_) + ".")
    text = ("> " + "\n>\n> ".join(parts) + "\n") if parts else ""
    return Leaf(f"auto.alert_{pair}", text), used


V21_DATA = HERE.parent / "package_v21" / "data_v21.json"   # published v2.1 campaign numbers, read-only
# (row label, data_v21.json model key or "", v3-release model key or "")
XVER_ROWS = [("Original Inkling 2 Mini", "orig", "stock"), ("{prime}", "", "prime"),
             ("Unbound v1.0", "v1", ""), ("Unbound v2", "v2", ""), ("Unbound v2.1", "v21", "v21"),
             ("**Unbound v3**", "", "v3"), ("IndrasMirror XQ", "xq", ""), ("AtomicChat Uncensored, LoRA", "atomic_lora", ""),
             ("AtomicChat Uncensored, AD-Q4_K file", "atomic_adq4", ""), ("OpenUncensored", "ou", ""),
             ("JoaoZaokk Heretic", "heretic", "")]


def _ci_pct(cell: dict | None) -> str:
    if not cell or cell.get("value") is None:
        return ""
    ci = cell.get("ci")
    return f"{cell['value']:.1f}%" + (f" [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "")


def v21_cells(key: str, data: dict) -> list[str]:
    """Columns measured by the v2.1 release (same campaign, n = 263): useful off, useful on, refused off / on,
    drift, MMLU-300 / GSM8K-100. Blank where that release did not measure the model."""
    if not key:
        return [""] * 4
    m, st = data["models"].get(key, {}), data["static"].get(key, {})
    ref = ""
    if m.get("refusal_off") and m.get("refusal_e09"):
        ref = f"{m['refusal_off']['value']:.1f}% / {m['refusal_e09']['value']:.1f}%"
    cap = "" if st.get("mmlu300") is None else f"{st['mmlu300']:.1f} / {st['gsm8k100']:.0f}"
    return [_ci_pct(m.get("useful_off")), _ci_pct(m.get("useful_e09")), ref, cap]


def v3_cells(model: str, vals: dict[str, Leaf]) -> tuple[list[str], list[str]]:
    """Columns measured by this release: uncensoredbench willing, hard debugging solved, MMLU (independent run)."""
    if not model:
        return [""] * 3, []
    pair = "prime" if model in ("stock", "prime") else "v3"
    group = "prime_vs_stock" if pair == "prime" else "v3_vs_v21"
    side = "base" if model in ("stock", "v21") else "fixed"
    keys = [f"ubench.willing_{model}", f"{group}.debug_solved_{model}", f"indep.mmlu_{pair}_{side}"]
    out = []
    for k in keys:
        leaf = vals[k]
        out.append("" if leaf.value is None else fmt(leaf) + ("†" if leaf.provisional else ""))
    return out, keys


def xver_table(vals: dict[str, Leaf]) -> tuple[Leaf, list[str]]:
    data = json.loads(V21_DATA.read_text())
    n_v21 = data["_meta"]["n"]
    n_ub, n_dbg = vals["ubench.n"], vals["prime_vs_stock.debug_n_tasks"]
    n_mm = vals["indep.mmlu_n"]
    # Sample sizes are stated once in the card text under the table, so the headers stay readable.
    head = ["Model", "Useful, thinking off (v2.1 release)", "Useful, thinking on (v2.1 release)",
            "Refused, off / on (v2.1 release)", "MMLU-300 / GSM8K-100 (v2.1 release)",
            "Drift from the original (v3 release)", "Sensitive requests: willing (v3 release)",
            "Hard debugging solved (v3 release)", "MMLU, 2,000 questions (v3 release)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    used = [n_ub.key, n_dbg.key, n_mm.key]
    drift_key = {"orig": "stock", "v1": "v1", "v2": "v2", "v21": "v21"}
    for label, k21, k3 in XVER_ROWS:
        cells3, keys = v3_cells(k3, vals)
        used += keys
        dk = f"drift.kld_{k3 or drift_key.get(k21, '')}"
        dleaf = vals.get(dk) if (k3 or k21 in drift_key) else None
        if dleaf is None:
            dcell = ""
        elif dleaf.value is None:
            dcell = IN_PROGRESS_TEXT if in_progress(dk) else ""
            used.append(dk)
        else:
            dcell = fmt(dleaf)
            used.append(dk)
        name = label.format(prime=vals["names.prime_title"].value)
        lines.append("| " + " | ".join([name, *v21_cells(k21, data), dcell, *cells3]) + " |")
    return Leaf("auto.xver_table", "\n".join(lines)), used


SEEDS = HERE / "seeds.json"   # schema in the file's _schema field


def pool_table(vals: dict[str, Leaf], pair: str) -> Leaf:
    """Pooled seeds (pool_<pair>.json via collect): passes / runs, paired difference and verdict. Empty if absent."""
    base, fixed = ("Original", str(vals["names.prime_name"].value)) if pair == "prime" else ("v2.1", "v3")
    rows = []
    for label, k in POOLED:
        get = lambda x: vals.get(f"pool.{k}_{pair}_{x}")  # noqa: E731
        if get("diff") is None or get("diff").value is None:
            continue
        rows.append(f"| {label} | {fmt(get('n'))} × {fmt(get('seeds'))} | {fmt(get('base'))} | {fmt(get('fixed'))} | "
                    f"{fmt(get('diff'))} | {fmt(get('verdict'))} |")
    if not rows:
        return Leaf(f"auto.pool_{pair}", "")
    head = (f"**Seeds pooled** (every item run once per seed; paired difference with a bootstrap that resamples items "
            f"as clusters, 10,000 resamples):\n\n| Benchmark | Items × seeds | {base} | {fixed} | Difference, points | "
            "Read-out |\n|---|---|---|---|---|---|")
    return Leaf(f"auto.pool_{pair}", head + "\n" + "\n".join(rows) + "\n")


def seed_table(vals: dict[str, Leaf]) -> Leaf:
    """Score per seed for each (benchmark, model, stack) in seeds.json; blank where a seed has not run."""
    if not SEEDS.exists():
        return Leaf("auto.seed_table", "")
    runs = json.loads(SEEDS.read_text())["runs"]
    names = {"stock": "Original", "prime": str(vals["names.prime_name"].value), "v21": "Unbound v2.1", "v3": "Unbound v3"}
    stacks = {"gpu": "our GPU server, released weights", "tinker": "Thinking Machines' hosted service, adapter not merged"}
    groups: dict[tuple[str, str, str], dict[int, dict]] = {}
    for r in runs:
        groups.setdefault((r["benchmark"], r["model"], r["stack"]), {})[int(r["seed"])] = r
    seeds = sorted({int(r["seed"]) for r in runs})
    lines = ["| Benchmark | Model | Setup | n | " + " | ".join(f"Seed {x}" for x in seeds) + " | Spread, points |",
             "|" + "---|" * (5 + len(seeds))]
    for (bench, model, stack), by in groups.items():
        cells = [f"{by[x]['score']:.1f}%" if x in by else "" for x in seeds]
        scores = [r["score"] for r in by.values()]
        spread = f"{max(scores) - min(scores):.1f}" if len(scores) > 1 else ""
        n = next(iter(by.values()))["n"]
        lines.append(f"| {bench} | {names.get(model, model)} | {stacks.get(stack, stack)} | {n} | "
                     + " | ".join(cells) + f" | {spread} |")
    return Leaf("auto.seed_table", "\n".join(lines))


def derived(vals: dict[str, Leaf]) -> tuple[dict[str, Leaf], list[str]]:
    """Values computed from others. Returns the leaves and the source keys they depend on."""
    name = vals["names.prime_name"]
    prime_title = f"Inkling-2-Mini-{name.value}"
    d = {
        "names.prime_title": Leaf("names.prime_title", prime_title, provisional=name.provisional),
        "names.prime_repo": Leaf("names.prime_repo", f"nokeepai/{prime_title}", provisional=name.provisional),
        "names.prime_lora_title": Leaf("names.prime_lora_title", f"{prime_title}-LoRA", provisional=name.provisional),
        "names.prime_lora_repo": Leaf("names.prime_lora_repo", f"nokeepai/{prime_title}-LoRA",
                                      provisional=name.provisional),
        "names.prime_gguf_title": Leaf("names.prime_gguf_title", f"{prime_title}-GGUF", provisional=name.provisional),
        "names.prime_gguf_repo": Leaf("names.prime_gguf_repo", f"nokeepai/{prime_title}-GGUF",
                                      provisional=name.provisional),
        # File names as scripts/v3rel/golive_v3.py uploads them: <prime_title>-<TYPE>.gguf
        "names.prime_gguf_file": Leaf("names.prime_gguf_file", f"{prime_title}-Q4_K_M.gguf",
                                      provisional=name.provisional),
    }
    d["names.prime_slug"] = Leaf("names.prime_slug", str(name.value).lower(), provisional=name.provisional)
    d["flags.name_is_honed"] = Leaf("flags.name_is_honed", str(name.value) != "Prime", unit="bool")
    for pair in ("prime", "v3"):
        diff = vals.get(f"ubench.delivery_paired_{pair}")
        word = None
        if diff is not None and diff.value is not None and diff.ci is not None:
            word = ("beyond noise (whole interval below zero)" if diff.ci[1] < 0 else
                    "beyond noise (whole interval above zero)" if diff.ci[0] > 0 else "within noise")
        d[f"auto.delivery_noise_{pair}"] = Leaf(f"auto.delivery_noise_{pair}", word)
    deps = [name.key]
    for pair, base_name in (("prime", "the original Inkling 2 Mini"), ("v3", "Unbound v2.1")):
        text, used = worse_list(vals, pair, base_name)
        deps += used
        d[f"auto.worse_{pair}"] = Leaf(f"auto.worse_{pair}", text)
    for pair in UB_PAIRS:
        d[f"auto.ubench_scope_{pair}"] = ubench_scope(vals, pair)
        d[f"auto.ubench_limit_{pair}"] = ubench_limit(vals, pair)
    vals_named = {**vals, **d}
    for pair, base_name in (("prime", "the original"), ("v3", "v2.1")):
        leaf, used = alert(vals, pair, base_name)
        d[leaf.key] = leaf
        deps += used
    d["auto.seed_table"] = seed_table(vals)
    for pair in ("prime", "v3"):
        d[f"auto.pool_{pair}"] = pool_table(vals, pair)
    leaf, used = xver_table(vals_named)
    d[leaf.key] = leaf
    deps += used
    return d, deps


# ---- rendering -------------------------------------------------------------------
IN_PROGRESS_TEXT = "run in progress, updated when it lands"
IN_PROGRESS: list[str] = []   # key prefixes from values.json meta.in_progress (set in main)


def in_progress(key: str) -> bool:
    """A TBD leaf whose run is genuinely still going: rendered as a marker, allowed by --final."""
    return any(key == k or key.startswith(k) for k in IN_PROGRESS)


@dataclass
class Report:
    missing: set[str]
    tbd: set[str]
    provisional: set[str]


def apply_ifs(text: str, vals: dict[str, Leaf], rep: Report) -> str:
    def pick(m: re.Match[str]) -> str:
        leaf = vals.get(m.group(1))
        if leaf is None:
            rep.missing.add(m.group(1))
            return ""
        return m.group(2) if leaf.value is True else (m.group(3) or "")
    return IF_BLOCK.sub(pick, text)


def fill(text: str, vals: dict[str, Leaf], rep: Report, draft: bool) -> str:
    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        leaf = vals.get(key)
        if leaf is None:
            rep.missing.add(key)
            return f"[MISSING {key}]"
        if leaf.value is None:
            if in_progress(key):
                return IN_PROGRESS_TEXT
            rep.tbd.add(key)
            return f"[TBD {key}]"
        if leaf.provisional:
            rep.provisional.add(key)
            return fmt(leaf) + ("†" if draft else "")
        return plain(fmt(leaf)) if m.group(2) else fmt(leaf)
    return PLACEHOLDER.sub(sub, text)


def chart_urls(text: str, repo: str, rev: str) -> str:
    # Hugging Face resolves relative images against main; point at the branch the charts are uploaded to.
    return text.replace("](charts/", f"]({HF}/{repo}/resolve/{rev}/charts/")


DRAFT_NOTE = ("> **DRAFT RENDER.** Values marked † are provisional (measured on a candidate before the final "
              "adapter). [TBD …] marks values not yet measured. Do not publish this file.\n\n")


def add_draft_note(text: str) -> str:
    if text.startswith("---\n"):
        end = text.index("\n---\n", 4) + 5
        return text[:end] + "\n" + DRAFT_NOTE + text[end:].lstrip("\n")
    return DRAFT_NOTE + text


# (template, dist folder, repo whose charts/ the card's images point at). Folder names are the HF repo names
# (scripts/v3rel/stage_v3.py reads dist/Inkling-2-Mini-<prime_name>[-LoRA|-GGUF]/README.md and
# dist/Inkling-2-Mini-Unbound/README.md), except github, which is the README of the GitHub repo.
def outputs(vals: dict[str, Leaf]) -> list[tuple[str, str, str]]:
    unbound = str(vals["names.unbound_repo"].value)
    prime = str(vals["names.prime_repo"].value)
    lora = str(vals["names.prime_lora_repo"].value)
    return [("main_README.md", unbound.split("/")[1], unbound),
            ("gguf_README.md", unbound.split("/")[1] + "-GGUF", unbound),
            ("basalt_README.md", unbound.split("/")[1] + "-Basalt", unbound),
            ("prime_README.md", prime.split("/")[1], prime),
            ("prime_lora_README.md", lora.split("/")[1], prime),
            ("prime_gguf_README.md", prime.split("/")[1] + "-GGUF", prime),
            # the adapter folder inside the Unbound repo (uploaded with the v3 files; owner layout 2026-11-27)
            ("lora_v3_fix_README.md", unbound.split("/")[1] + "/lora-v3-fix", unbound),
            ("github_README.md", "github", unbound)]


def launch_outputs(vals: dict[str, Leaf]) -> list[tuple[Path, Path, str]]:
    """Launch drafts (posts the owner publishes himself): launch/<name>.md -> dist/launch/<name>.md."""
    return [(f, Path("launch") / f.name, str(vals["names.unbound_repo"].value))
            for f in sorted((HERE / "launch").glob("*.md"))]


def note_dependencies(text: str, vals: dict[str, Leaf], deps: list[str], rep: Report) -> None:
    """An auto.* list built from TBD or provisional inputs is itself not final."""
    if "{{auto." not in text:
        return
    for k in deps:
        leaf = vals[k]
        if leaf.value is None and in_progress(k):
            continue
        if leaf.value is None:
            rep.tbd.add(k)
        elif leaf.provisional:
            rep.provisional.add(k)


def render_one(template: Path, vals: dict[str, Leaf], deps: list[str], repo: str, draft: bool,
               rep: Report) -> str:
    text = template.read_text()
    note_dependencies(text, vals, deps, rep)
    text = fill(apply_ifs(text, vals, rep), vals, rep, draft)
    text = chart_urls(text, repo, str(vals["meta.chart_rev"].value))
    return add_draft_note(text) if draft else text


def print_report(rep: Report, final: bool) -> int:
    if rep.missing:
        print("ERROR: placeholders with no value in values.json (typo or missing key):")
        print("  " + "\n  ".join(sorted(rep.missing)))
    print(f"TBD used ({len(rep.tbd)}):", " ".join(sorted(rep.tbd)) or "none")
    print(f"provisional used ({len(rep.provisional)}):", " ".join(sorted(rep.provisional)) or "none")
    if rep.missing:
        return 1
    if final and (rep.tbd or rep.provisional):
        print("ERROR: --final needs every used value measured. Run collect_v3.py, fill the rest by hand, "
              "and clear 'provisional' only on values you have checked.")
        return 1
    return 0


def warn_name(vals: dict[str, Leaf]) -> None:
    """Owner rule: 'Prime' only if equal or better on every check. Lists exactly the checks whose paired interval is
    entirely below zero; the decision taken is recorded in names.prime_name_decision."""
    rows = beyond_noise_labels(vals, "prime")
    pooled = [k for k, leaf in vals.items() if k.startswith("pool.") and k.endswith("_prime_verdict")
              and leaf.value == "WORSE"]
    if (rows or pooled) and vals["names.prime_name"].value == "Prime":
        print("WARNING: below stock beyond noise: " + "; ".join(rows + pooled) + ". Owner rule: 'Prime' only if "
              "equal or better on every check; render with PRIME_NAME=Honed, or record the decision in "
              "names.prime_name_decision.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="fail if any used value is TBD or provisional")
    ap.add_argument("--draft", action="store_true", help="default; TBD and provisional values marked visibly")
    ap.add_argument("--values", default=str(HERE / "values.json"))
    ap.add_argument("--out", default=str(HERE / "dist"))
    a = ap.parse_args()
    vals = load_values(Path(a.values)) | load_v21_card()
    ip = vals.get("meta.in_progress")
    IN_PROGRESS[:] = list(ip.value) if ip and isinstance(ip.value, list) else []
    extra, deps = derived(vals)
    vals.update(extra)
    rep = Report(set(), set(), set())
    jobs = [(HERE / "templates" / t, Path(folder) / "README.md", repo) for t, folder, repo in outputs(vals)]
    jobs += launch_outputs(vals)
    # The changelog and release notes live at package_v3/ itself, except in test renders to another --out.
    docs_dir = HERE if Path(a.out) == HERE / "dist" else Path(a.out)
    jobs += [(HERE / "templates" / n, docs_dir / n, str(vals["names.unbound_repo"].value))
             for n in ("CHANGELOG.md", "RELEASE_NOTES.md")]
    rendered = {dest: render_one(src, vals, deps, repo, not a.final, rep) for src, dest, repo in jobs}
    status = print_report(rep, a.final)
    if status == 0:
        for rel, text in rendered.items():
            dest = Path(a.out) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
            print("wrote", dest)
    warn_name(vals)
    return status


if __name__ == "__main__":
    sys.exit(main())
