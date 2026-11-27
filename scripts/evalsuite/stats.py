"""Confidence intervals and paired tests on saved per-item results; numbers only, never answer text.

  python3 stats.py TAG [TAG...] --ref TAG [--out-root ~/.inkling-abl/evalsuite] [--suites pub_gsm8k,...]
      [--margin 2] [--json PATH] [--repeat 0] [--resamples 10000] [--seed 1234]
  (the same report: python3 compare.py TAGS... --ci --ref TAG ...)

For each suite x metric x tag: n, score and a Wilson 95% interval. Against --ref, on the items both tags answered
(no request error): the paired difference in points, a 95% paired-bootstrap interval over items, the exact McNemar
p-value, and a flag when the two tags' item sets differ. Pooled lines per family (coding, bug-fixing, maths,
knowledge) count each public item once. Non-inferiority reads the bootstrap lower bound against -margin points;
a row whose whole interval is below zero says WORSE, so a real loss cannot hide behind "inconclusive".
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics as M  # noqa: E402

Rec = dict[str, Any]
Outcome = Callable[[Rec], "bool | None"]   # None = the item has no score for this metric (main's non-maths rows)


def _flag(key: str) -> Outcome:
    return lambda r: bool(r[key])


def _strict(r: Rec) -> bool | None:
    return None if r.get("gold") is None else bool(r["maths_correct"])


def _number(r: Rec) -> bool | None:
    """Recomputed from the saved answer, so results written before accuracy_number existed get it too."""
    return None if r.get("gold") is None else bool(M.maths_number_quality(r.get("answer") or "", r["gold"]))


_CODE = [("pass@1", _flag("passed"))]
_MATHS = [("strict", _strict), ("number", _number)]
METRICS: dict[str, list[tuple[str, Outcome]]] = {
    "pub_humaneval": _CODE, "pub_mbpp": _CODE, "pub_quixbugs": _CODE, "pub_humanevalfix": _CODE, "code": _CODE,
    "pub_gsm8k": _MATHS, "pub_gsm8k_full": _MATHS, "hardmath": _MATHS, "main": _MATHS,
    "pub_mmlu": [("accuracy", _flag("correct"))], "pub_mmlu_2k": [("accuracy", _flag("correct"))],
    "pub_math500": [("is_equiv", _flag("correct"))],
}
# Larger suite first: pub_gsm8k's items are a subset of pub_gsm8k_full's (same ids) and pub_mmlu shares ids with
# pub_mmlu_2k, so pooling takes each id once, from the first suite in which both tags have it.
FAMILIES: list[tuple[str, list[str], list[str]]] = [
    ("coding", ["pub_humaneval", "pub_mbpp"], ["pass@1"]),
    ("bug-fixing", ["pub_humanevalfix", "pub_quixbugs"], ["pass@1"]),
    ("maths", ["pub_gsm8k_full", "pub_gsm8k", "pub_math500"], ["strict", "number"]),
    ("knowledge", ["pub_mmlu_2k", "pub_mmlu"], ["accuracy"]),
]
DEFAULT_SUITES = ("pub_humaneval", "pub_quixbugs", "pub_gsm8k", "pub_mmlu", "pub_gsm8k_full", "pub_mmlu_2k",
                  "pub_mbpp", "pub_humanevalfix", "pub_math500")


@dataclass
class Items:
    ok: dict[str, Rec]     # answered items (no request error) by id
    all_ids: set[str]      # every id in the file, errors included
    errors: int

# ---------- statistics ----------


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: discordant pairs b (ref right only) and c (candidate right only) under
    Binomial(b + c, 1/2); p = 2 x the smaller tail, capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def _binomial(rng: random.Random, n: int, p: float) -> int:
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    if hasattr(rng, "binomialvariate"):  # Python 3.12+
        return rng.binomialvariate(n, p)
    return sum(rng.random() < p for _ in range(n))


def quantile(xs: list[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default), xs sorted."""
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap_diff(b: int, c: int, n: int, resamples: int, seed: int) -> tuple[float, float]:
    """95% percentile interval of mean(candidate - ref) under a paired bootstrap over items. Each per-item
    difference is -1 (b items), +1 (c items) or 0, so resampling n items with replacement is exactly a multinomial
    draw of those three counts; drawing the counts directly is the same bootstrap, in milliseconds."""
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        up = _binomial(rng, n, c / n)
        down = _binomial(rng, n - up, b / (n - c)) if n > c else 0
        draws.append((up - down) / n)
    draws.sort()
    return quantile(draws, 0.025), quantile(draws, 0.975)


def single(flags: dict[str, bool]) -> Rec:
    k, n = sum(flags.values()), len(flags)
    lo, hi = M.wilson(k, n)
    return {"n": n, "k": k, "score": k / n if n else 0.0, "wilson_lo": lo, "wilson_hi": hi}


def verdict(lo: float, hi: float, margin: float) -> str:
    if hi < 0:
        return "WORSE"
    if lo >= -margin:
        return "non-inferior" + (", better" if lo > 0 else "")
    return "inconclusive"


def paired(ref: dict[str, bool], cand: dict[str, bool], margin: float, resamples: int, seed: int) -> Rec:
    ids = [i for i in ref if i in cand]
    b = sum(ref[i] and not cand[i] for i in ids)
    c = sum(cand[i] and not ref[i] for i in ids)
    n = len(ids)
    lo, hi = bootstrap_diff(b, c, n, resamples, seed)
    diff = (c - b) / n if n else 0.0
    return {"n_paired": n, "ref_only_right": b, "tag_only_right": c, "diff": diff, "boot_lo": lo, "boot_hi": hi,
            "mcnemar_p": mcnemar_exact(b, c), "non_inferior": n > 0 and lo >= -margin,
            "verdict": verdict(lo, hi, margin) if n else "no shared items"}

# ---------- loading ----------


def load_items(root: Path, tag: str, suite: str, repeat: int) -> Items | None:
    path = root / tag / suite / f"r{repeat}.jsonl"
    if not path.exists():
        return None
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    ok = {str(r["id"]): r for r in recs if not r.get("error")}
    return Items(ok=ok, all_ids={str(r["id"]) for r in recs}, errors=len(recs) - len(ok))


def scored(items: Items, fn: Outcome) -> dict[str, bool]:
    out = {}
    for i, r in items.ok.items():
        v = fn(r)
        if v is not None:
            out[i] = v
    return out


def match_info(ref: Items, cand: Items) -> Rec:
    only_ref, only_tag = len(ref.all_ids - cand.all_ids), len(cand.all_ids - ref.all_ids)
    same = only_ref == 0 and only_tag == 0 and ref.errors == 0 and cand.errors == 0
    return {"items_match": same, "only_ref": only_ref, "only_tag": only_tag, "errors_ref": ref.errors,
            "errors_tag": cand.errors}

# ---------- report rows ----------


@dataclass
class Settings:
    root: Path
    tags: list[str]
    ref: str | None
    suites: list[str]
    margin: float      # as a fraction (2 points = 0.02)
    repeat: int
    resamples: int
    seed: int


def suite_rows(st: Settings, suite: str, loaded: dict[str, Items]) -> list[Rec]:
    rows = []
    for metric, fn in METRICS[suite]:
        flags = {t: scored(it, fn) for t, it in loaded.items()}
        for t in st.tags:
            if t not in loaded:
                continue
            row: Rec = {"suite": suite, "metric": metric, "tag": t, **single(flags[t])}
            if st.ref in loaded and t != st.ref:
                row["vs_ref"] = {**paired(flags[st.ref], flags[t], st.margin, st.resamples, st.seed),
                                 **match_info(loaded[st.ref], loaded[t])}
            rows.append(row)
    return rows


def pooled_flags(per_suite: list[dict[str, bool]]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for flags in per_suite:
        for i, v in flags.items():
            out.setdefault(i, v)
    return out


def metric_fn(suite: str, metric: str) -> Outcome:
    named = dict(METRICS[suite])
    return named.get(metric, METRICS[suite][0][1])  # math500 has one reading; it serves strict and number


def family_pair(st: Settings, suites: list[str], metric: str, data: dict[str, dict[str, Items]],
                tag: str) -> tuple[dict[str, bool], dict[str, bool]]:
    """Ref and tag outcomes pooled over a family, each id once, from the first suite where both tags have it."""
    ref_out: dict[str, bool] = {}
    tag_out: dict[str, bool] = {}
    for s in suites:
        if tag not in data[s] or st.ref not in data[s]:
            continue
        fn = metric_fn(s, metric)
        r, t = scored(data[s][st.ref], fn), scored(data[s][tag], fn)
        for i in r.keys() & t.keys():
            if i not in ref_out:
                ref_out[i], tag_out[i] = r[i], t[i]
    return ref_out, tag_out


def family_rows(st: Settings, data: dict[str, dict[str, Items]]) -> list[Rec]:
    rows = []
    for fam, members, metrics in FAMILIES:
        suites = [s for s in members if s in data]
        for metric in metrics:
            for t in st.tags:
                own = [scored(data[s][t], metric_fn(s, metric)) for s in suites if t in data[s]]
                if not own:
                    continue
                row: Rec = {"family": fam, "metric": metric, "tag": t,
                            "suites": [s for s in suites if t in data[s]], **single(pooled_flags(own))}
                if st.ref and t != st.ref and any(st.ref in data[s] and t in data[s] for s in suites):
                    ref_out, tag_out = family_pair(st, suites, metric, data, t)
                    row["vs_ref"] = paired(ref_out, tag_out, st.margin, st.resamples, st.seed)
                rows.append(row)
    return rows

# ---------- output ----------


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def vs_cells(row: Rec, ref: str | None) -> list[str]:
    v = row.get("vs_ref")
    if v is None:
        return ["ref" if row["tag"] == ref else "–"] + [""] * 5
    items = "match" if v.get("items_match", True) else (
        f"MISMATCH (only ref {v['only_ref']}, only tag {v['only_tag']}, errors {v['errors_ref']}/{v['errors_tag']})")
    if "items_match" not in v:
        items = f"{v['n_paired']} paired"
    return [pts(v["diff"]), f"[{pts(v['boot_lo'])}, {pts(v['boot_hi'])}]", f"{v['mcnemar_p']:.3g}",
            f"{v['ref_only_right']}/{v['tag_only_right']}", items, v["verdict"]]


def markdown(st: Settings, rows: list[Rec], fams: list[Rec]) -> str:
    m = f"{100 * st.margin:g}"
    head = (f"| {{}} | metric | tag | n | score | 95% Wilson | Δ vs {st.ref or '–'} (pts) | 95% paired bootstrap | "
            f"McNemar p | ref-only/tag-only right | items | non-inferiority (lower ≥ −{m} pts) |")
    sep = "|---" * 12 + "|"
    out = [f"### per suite (repeat r{st.repeat}; {st.resamples:,} resamples, seed {st.seed})", "",
           head.format("suite"), sep]
    for r in rows:
        out.append("| " + " | ".join([r["suite"], r["metric"], r["tag"], str(r["n"]), pct(r["score"]),
                                       f"[{pct(r['wilson_lo'])}, {pct(r['wilson_hi'])}]", *vs_cells(r, st.ref)])
                   + " |")
    out += ["", "### pooled by family (each item once)", "", head.format("family"), sep]
    for r in fams:
        name = f"{r['family']} ({'+'.join(x.removeprefix('pub_') for x in r['suites'])})"
        out.append("| " + " | ".join([name, r["metric"], r["tag"], str(r["n"]), pct(r["score"]),
                                       f"[{pct(r['wilson_lo'])}, {pct(r['wilson_hi'])}]", *vs_cells(r, st.ref)])
                   + " |")
    return "\n".join(out) + "\n"


def report(st: Settings) -> tuple[str, Rec]:
    data: dict[str, dict[str, Items]] = {}
    for s in st.suites:
        loaded = {t: it for t in st.tags if (it := load_items(st.root, t, s, st.repeat)) is not None}
        if loaded:
            data[s] = loaded
    rows = [r for s in st.suites if s in data for r in suite_rows(st, s, data[s])]
    fams = family_rows(st, data)
    js = {"ref": st.ref, "tags": st.tags, "margin_points": 100 * st.margin, "repeat": st.repeat,
          "resamples": st.resamples, "seed": st.seed, "rows": rows, "families": fams}
    return markdown(st, rows, fams), js


def add_args(ap: argparse.ArgumentParser) -> None:
    """The options compare.py --ci shares with this script."""
    ap.add_argument("--ref", default=None, help="reference tag for paired differences")
    ap.add_argument("--margin", type=float, default=2.0, help="non-inferiority margin in points (default 2)")
    ap.add_argument("--json", default=None, help="write the numbers (no text) to this JSON file")
    ap.add_argument("--repeat", type=int, default=0, help="which r<i>.jsonl to read")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)


def run(tags: list[str], out_root: str, suites: list[str], a: argparse.Namespace) -> str:
    st = Settings(root=Path(out_root).expanduser(), tags=tags, ref=a.ref,
                  suites=[s for s in suites if s in METRICS], margin=a.margin / 100, repeat=a.repeat,
                  resamples=a.resamples, seed=a.seed)
    if st.ref and st.ref not in st.tags:
        st.tags = [st.ref, *st.tags]
    text, js = report(st)
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(js, indent=1))
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out-root", default="~/.inkling-abl/evalsuite")
    ap.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    add_args(ap)
    a = ap.parse_args()
    print(run(a.tags, a.out_root, a.suites.split(","), a))


def checks() -> None:
    """Hand-computed cases (selftest.py runs these)."""
    lo, hi = M.wilson(8, 10)  # p=.8, z=1.96: centre .99207/1.38415 = .71674, half .31362/1.38415 = .22658
    assert abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3, (lo, hi)
    lo, hi = M.wilson(10, 10)
    assert abs(lo - 0.7225) < 1e-3 and abs(hi - 1.0) < 1e-12
    assert abs(mcnemar_exact(1, 5) - 0.21875) < 1e-12        # 2 * (C(6,0) + C(6,1)) / 2^6
    assert abs(mcnemar_exact(0, 6) - 0.03125) < 1e-12        # 2 / 64
    assert mcnemar_exact(3, 3) == 1.0 and mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(2, 10) - 0.0385742) < 1e-6     # 2 * (1 + 12 + 66) / 4096
    assert bootstrap_diff(0, 0, 50, 1000, 1) == (0.0, 0.0)
    assert bootstrap_diff(0, 50, 50, 1000, 1) == (1.0, 1.0)
    lo, hi = bootstrap_diff(5, 15, 200, 10000, 1234)  # diff .05, sd of mean .0221; draws move in steps of .005
    assert 0.0 < lo <= 0.015 and 0.085 <= hi <= 0.1, (lo, hi)
    assert bootstrap_diff(5, 15, 200, 2000, 7) == bootstrap_diff(5, 15, 200, 2000, 7)
    assert quantile([0.0, 1.0, 2.0, 3.0], 0.5) == 1.5
    p = paired({"a": True, "b": True, "c": False, "d": True}, {"a": True, "b": False, "c": True, "e": True},
               0.02, 500, 1)
    assert p["n_paired"] == 3 and p["ref_only_right"] == 1 and p["tag_only_right"] == 1 and p["diff"] == 0.0
    assert verdict(-0.01, 0.03, 0.02) == "non-inferior" and verdict(-0.05, -0.01, 0.02) == "WORSE"
    assert verdict(-0.03, 0.02, 0.02) == "inconclusive" and verdict(0.01, 0.05, 0.02) == "non-inferior, better"
    assert pooled_flags([{"x": True}, {"x": False, "y": False}]) == {"x": True, "y": False}


if __name__ == "__main__":
    main()
