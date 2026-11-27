#!/usr/bin/env python3
"""Loader for package_v3/values.json: every number the v3 charts plot, plus which keys each chart uses.

Usage: python3 scripts/charts/data_v3.py [--values PATH]   (prints a status table of every key the charts use)
values.json is the only data source. A leaf is the string "TBD" or {value, unit, provisional?, display?, note?}.
Status: "measured" (a value, not provisional), "provisional" (measured on a candidate before the final adapter),
"tbd" (no value yet). The chart specs live here so make_v3.py and this table can never disagree about what is
plotted.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, NamedTuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
VALUES_PATH = os.path.join(ROOT, "package_v3", "values.json")

Status = Literal["measured", "provisional", "tbd"]
Leaf = Union[str, dict[str, object]]
Key = tuple[str, str]  # (group, key)
Pair = Literal["prime", "unbound"]


@dataclass(frozen=True)
class Val:
    value: float | None
    status: Status
    display: str | None
    unit: str
    text: str | None  # set for unit "text" leaves (names), where value stays None
    ci: tuple[float, float] | None = None  # 95% interval when the collector recorded one

    @property
    def is_tbd(self) -> bool:
        return self.status == "tbd"

    @property
    def is_provisional(self) -> bool:
        return self.status == "provisional"


class Row(NamedTuple):
    """One base-vs-fixed comparison row of the level chart."""
    label: str
    base: Key
    fixed: Key
    n: Key


@lru_cache(maxsize=1)
def load() -> dict[str, dict[str, Leaf]]:
    with open(VALUES_PATH) as f:
        return json.load(f)


def set_values_path(path: str) -> None:
    """Read another values file (a dry run's scratch copy) instead of package_v3/values.json."""
    global VALUES_PATH
    VALUES_PATH = os.path.abspath(os.path.expanduser(path))
    load.cache_clear()


def parse(leaf: Leaf) -> Val:
    if not isinstance(leaf, dict):
        return Val(None, "tbd", None, "", None)
    raw = leaf.get("value")
    unit = str(leaf.get("unit", ""))
    status: Status = "provisional" if leaf.get("provisional") else "measured"
    display = leaf.get("display")
    display_s = str(display) if display is not None else None
    if isinstance(raw, bool) or raw is None:
        return Val(None, status, display_s, unit, None if raw is None else str(raw))
    if isinstance(raw, (int, float)):
        return Val(float(raw), status, display_s, unit, None, parse_ci(leaf.get("ci")))
    return Val(None, status, display_s, unit, str(raw))


def parse_ci(raw: object) -> tuple[float, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and all(isinstance(x, (int, float)) for x in raw):
        return float(raw[0]), float(raw[1])
    return None


def get(group: str, key: str) -> Val:
    # KeyError on a missing key is deliberate: a typo must fail loudly, not draw as TBD.
    return parse(load()[group][key])


def gk(k: Key) -> Val:
    return get(k[0], k[1])


def has(group: str, key: str) -> bool:
    return key in load().get(group, {})


def prime_name() -> str:
    return get("names", "prime_name").text or "Prime"


def prime_title() -> str:
    # Same derivation as package_v3/render.py's names.prime_title, so the banner and the card agree.
    return f"Inkling-2-Mini-{prime_name()}"


def flag(key: str) -> bool:
    leaf = load()["flags"][key]
    return isinstance(leaf, dict) and leaf.get("value") is True


# ---- chart specs ------------------------------------------------------------
PAIR_GROUP: dict[Pair, str] = {"prime": "prime_vs_stock", "unbound": "v3_vs_v21"}
PAIR_SUFFIX: dict[Pair, tuple[str, str]] = {"prime": ("stock", "prime"), "unbound": ("v21", "v3")}
PAIRS: tuple[Pair, Pair] = ("prime", "unbound")
HERO_STEMS: tuple[tuple[str, str], ...] = (
    ("debug_solved", "Hard debugging solved"),
    ("debug_nocode", "Stalled turns with no code"),
    ("debug_tokens", "Tokens per task"),
)
SPEED_STEMS: tuple[tuple[str, str, str], ...] = (
    ("think_main", "Median thinking tokens", "main_n"),
    ("cap_ubench", "Thinking-cap hits, uncensored benchmark", "ubench_n"),
    ("sec_main", "Seconds per answer", "main_n"),
)
HISTORY_VERSIONS: tuple[tuple[str, str], ...] = (("v2", "Unbound v2"), ("v21", "Unbound v2.1"), ("v3", "Unbound v3"))
INDEP_BENCH: tuple[tuple[str, str], ...] = (
    ("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("humanevalfix", "HumanEvalFix"), ("quixbugs", "QuixBugs"),
    ("gsm8k_strict", "GSM8K, strict"), ("gsm8k_lenient", "GSM8K, number correct"),
    ("math500", "MATH-500, is_equiv"), ("math500_number", "MATH-500, number correct"), ("mmlu", "MMLU"),
)
INDEP_PAIR: dict[Pair, str] = {"prime": "prime", "unbound": "v3"}
INDEP_MARGIN: Key = ("indep", "margin_pts")
INDEP_DECODING: Key = ("indep", "decoding")
THINK_STATS: tuple[tuple[str, str], ...] = (
    ("median", "Median"), ("mean", "Mean"), ("p90", "90th percentile"), ("p99", "99th percentile"),
    ("nonterm", "Never finished thinking"),
)
THINK_MODELS: tuple[str, ...] = ("stock", "prime", "v21", "v3")
THINK_SETTINGS: Key = ("think_dist", "settings")
TOKEN_RATIO: Key = ("headline", "debug_token_ratio")
PRIME_NAME: Key = ("names", "prime_name")


def pair_key(pair: Pair, stem: str, fixed: bool) -> Key:
    return PAIR_GROUP[pair], f"{stem}_{PAIR_SUFFIX[pair][int(fixed)]}"


def flat_key(group: str, pair: Pair, stem: str, fixed: bool) -> Key:
    return group, f"{stem}_{PAIR_SUFFIX[pair][int(fixed)]}"


def speed_n_key(which: str) -> Key:
    return ("ubench", "n") if which == "ubench_n" else ("speed", "main_n")


def level_rows(pair: Pair) -> list[Row]:
    lab = PAIR_GROUP[pair]

    def labrow(label: str, stem: str, n: str) -> Row:
        return Row(label, pair_key(pair, stem, False), pair_key(pair, stem, True), (lab, n))

    def pubrow(label: str, stem: str, n: str) -> Row:
        g = "public_benchmarks"
        return Row(label, flat_key(g, pair, stem, False), flat_key(g, pair, stem, True), (g, n))

    return [
        labrow("Answer quality, independent grader", "quality_good", "quality_n"),
        labrow("Creative writing, good", "creative_good", "creative_n"),
        labrow("Roleplay, good", "rp_good", "rp_n"),
        labrow("Code, pass@1", "code_pass", "code_n"),
        labrow("Hard maths, strict", "maths_strict", "maths_n"),
        labrow("Hard maths, number correct", "maths_lenient", "maths_n"),
        pubrow("HumanEval, pass@1", "humaneval", "humaneval_n"),
        pubrow("QuixBugs, fixed", "quixbugs", "quixbugs_n"),
        pubrow("GSM8K, strict", "gsm8k_strict", "gsm8k_n"),
        pubrow("GSM8K, number correct", "gsm8k_lenient", "gsm8k_n"),
        pubrow("MMLU", "mmlu", "mmlu_n"),
        Row("Uncensored benchmark, delivery", flat_key("ubench", pair, "delivery", False),
            flat_key("ubench", pair, "delivery", True), ("ubench", f"n_{PAIR_SUFFIX[pair][1]}")),
    ]


def hero_keys() -> list[Key]:
    keys: list[Key] = [PRIME_NAME, (PAIR_GROUP["prime"], "debug_n_tasks"), (PAIR_GROUP["unbound"], "debug_n_tasks")]
    for pair in PAIRS:
        keys += [pair_key(pair, s, f) for s, _ in HERO_STEMS for f in (False, True)]
    if any(gk(pair_key(p, "debug_tokens", f)).is_tbd for p in PAIRS for f in (False, True)):
        keys.append(TOKEN_RATIO)
    return keys


def speed_keys() -> list[Key]:
    keys: list[Key] = [PRIME_NAME]
    for stem, _, nk in SPEED_STEMS:
        keys.append(speed_n_key(nk))
        keys += [flat_key("speed", p, stem, f) for p in PAIRS for f in (False, True)]
    return list(dict.fromkeys(keys))


def nothing_lost_keys(pair: Pair) -> list[Key]:
    keys: list[Key] = [PRIME_NAME] if pair == "prime" else []
    for r in level_rows(pair):
        keys += [r.base, r.fixed, r.n]
    return list(dict.fromkeys(keys))


def history_keys() -> list[Key]:
    """v2 slots are excluded on purpose: no v2 run exists, so they show 'not measured' and never block --final."""
    keys: list[Key] = [("ubench", "n_v3")]
    for v in ("v21", "v3"):
        keys += [("ubench", f"willing_{v}"), ("ubench", f"delivery_{v}"), ("speed", f"cap_ubench_{v}")]
    return keys


def indep_diff_key(pair: Pair, bench: str) -> Key:
    return "indep", f"{bench}_{INDEP_PAIR[pair]}_diff"


POOLED_BENCH: tuple[tuple[str, str], ...] = (("humanevalfix", "HumanEvalFix"), ("mbpp", "MBPP"))


def pooled_rows(pair: Pair) -> list[tuple[str, Val, Val, str | None]]:
    """Multi-seed pooled rows for the forest chart, present only when collect wrote them (pool_<pair>.json)."""
    p = INDEP_PAIR[pair]
    out = []
    for k, label in POOLED_BENCH:
        if not has("pool", f"{k}_{p}_diff") or get("pool", f"{k}_{p}_diff").is_tbd:
            continue
        seeds = get("pool", f"{k}_{p}_seeds")
        n = get("pool", f"{k}_{p}_n")
        runs = Val((n.value or 0) * (seeds.value or 0), "measured", None, "n", None)
        out.append((f"{label}, {seeds.value:.0f} seeds pooled", runs, get("pool", f"{k}_{p}_diff"),
                    get("pool", f"{k}_{p}_verdict").text))
    return out


def indep_keys(pair: Pair) -> list[Key]:
    keys: list[Key] = ([PRIME_NAME] if pair == "prime" else []) + [INDEP_MARGIN, INDEP_DECODING]
    for b, _ in INDEP_BENCH:
        keys += [("indep", f"{b}_{INDEP_PAIR[pair]}_n"), indep_diff_key(pair, b)]
    return keys


def thinkdist_keys() -> list[Key]:
    keys: list[Key] = [PRIME_NAME, THINK_SETTINGS, ("think_dist", "n")]
    return keys + [("think_dist", f"{s}_{m}") for s, _ in THINK_STATS for m in THINK_MODELS]


# Cross-version chart: the v2.1 release's published campaign (read-only) and this release's own measurements.
V21_DATA_PATH = os.path.join(ROOT, "package_v21", "data_v21.json")
XVER_V21_MODELS: tuple[tuple[str, str], ...] = (
    ("orig", "Original"), ("v1", "Unbound\nv1.0"), ("v2", "Unbound\nv2"), ("v21", "Unbound\nv2.1"),
    ("xq", "Indras-\nMirror XQ"), ("atomic_lora", "AtomicChat\nLoRA"), ("atomic_adq4", "AtomicChat\nAD-Q4_K"),
    ("ou", "Open-\nUncensored"), ("heretic", "Heretic"))
XVER_V3_MODELS: tuple[str, ...] = ("stock", "prime", "v21", "v3")


@lru_cache(maxsize=1)
def v21_data() -> dict:
    with open(V21_DATA_PATH) as f:
        return json.load(f)


def xver_keys() -> list[Key]:
    keys: list[Key] = [PRIME_NAME, ("ubench", "n"), (PAIR_GROUP["prime"], "debug_n_tasks")]
    keys += [("ubench", f"willing_{m}") for m in XVER_V3_MODELS]
    keys += [pair_key("prime", "debug_solved", f) for f in (False, True)]
    keys += [pair_key("unbound", "debug_solved", f) for f in (False, True)]
    keys.append(USEFUL263_FLAG)
    if flag(USEFUL263_FLAG[1]):
        keys.append(("useful263", "v3_on"))
    return keys


# Banners: the 263-request usefulness keys only count once that re-run is done; until then they read "pending"
# and the Unbound banner shows the uncensored-benchmark answer rate instead.
USEFUL263_FLAG: Key = ("flags", "useful263_done")


def reddit_useful_keys() -> list[Key]:
    """Reddit 'is unbound still uncensored?' chart: built only once the 263-request re-run is done.
    The other rows come from v2.1's published card values (v21_card), which are not values.json keys."""
    return [USEFUL263_FLAG] + [("useful263", f"{m}_{t}") for m in ("v3", "v21") for t in ("off", "on")]


def banner_keys() -> list[Key]:
    keys: list[Key] = [USEFUL263_FLAG]
    if flag(USEFUL263_FLAG[1]):
        keys += [("useful263", "v3_off"), ("useful263", "v21_off")]
    else:
        keys += [("ubench", f"willing_{m}") for m in ("v3", "v21", "stock")]
    keys += [pair_key("unbound", "debug_solved", f) for f in (True, False)]
    keys += [flat_key("speed", "unbound", "think_main", f) for f in (True, False)]
    return keys


def banner_prime_keys() -> list[Key]:
    keys: list[Key] = [PRIME_NAME]
    keys += [pair_key("prime", s, f) for s in ("debug_solved", "debug_nocode") for f in (True, False)]
    keys += [flat_key("speed", "prime", "think_main", f) for f in (True, False)]
    return keys


def reddit_keys() -> list[Key]:
    """Reddit image: Prime and Unbound v3 side by side, three shared before/after cells and one 'kind of model' cell."""
    keys: list[Key] = [PRIME_NAME, USEFUL263_FLAG, (PAIR_GROUP["prime"], "debug_n_tasks"), ("speed", "main_n")]
    for pair in PAIRS:
        keys += [pair_key(pair, "debug_solved", f) for f in (False, True)]
        keys += [flat_key("speed", pair, s, f) for s in ("think_main", "sec_main") for f in (False, True)]
    keys += [("ubench", "willing_stock"), ("ubench", "willing_prime")]
    if flag(USEFUL263_FLAG[1]):
        keys += [("useful263", "v3_on"), ("useful263", "v21_on"), ("useful263", "bison_n_on")]
    return keys


V21_CARD_PATH = os.path.join(ROOT, "package_v21", "values.json")


@lru_cache(maxsize=1)
def v21_card() -> dict[str, str]:
    """v2.1's published card values (render.py exposes them as v21card.KEY), read-only: KEY -> display string."""
    with open(V21_CARD_PATH) as f:
        return json.load(f)


def chart_keys() -> dict[str, list[Key]]:
    useful: dict[str, list[Key]] = {"reddit_useful_v3": reddit_useful_keys()} if flag(USEFUL263_FLAG[1]) else {}
    return useful | {
        "banner_v3": banner_keys(),
        "banner_prime_v3": banner_prime_keys(),
        "reddit_v3": reddit_keys(),
        "xver_v3": xver_keys(),
        "hero_v3": hero_keys(),
        "speed_v3": speed_keys(),
        "nothing_lost_prime_v3": nothing_lost_keys("prime"),
        "nothing_lost_unbound_v3": nothing_lost_keys("unbound"),
        "history_v3": history_keys(),
        "indep_prime_v3": indep_keys("prime"),
        "indep_unbound_v3": indep_keys("unbound"),
        "thinkdist_v3": thinkdist_keys(),
    }


def all_chart_keys() -> list[Key]:
    return list(dict.fromkeys(k for ks in chart_keys().values() for k in ks))


def not_final(keys: list[Key]) -> list[Key]:
    return [k for k in keys if gk(k).status != "measured"]


# ---- status table -----------------------------------------------------------
def fmt_value(v: Val) -> str:
    if v.text is not None:
        return v.text
    if v.value is None:
        return "TBD"
    return f"{v.value:g} {v.unit}".strip()


def main(argv: list[str]) -> int:
    if "--values" in argv:
        set_values_path(argv[argv.index("--values") + 1])
    keys = all_chart_keys()
    width = max(len(f"{g}.{k}") for g, k in keys)
    print(f"{'key':<{width}}  {'value':<16}  status")
    for g, k in keys:
        v = get(g, k)
        print(f"{g + '.' + k:<{width}}  {fmt_value(v):<16}  {v.status}")
    counts = {s: sum(get(g, k).status == s for g, k in keys) for s in ("measured", "provisional", "tbd")}
    print(f"\n{len(keys)} keys: " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
