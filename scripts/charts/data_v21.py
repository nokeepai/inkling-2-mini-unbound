#!/usr/bin/env python3
"""Build package_v21/data_v21.json: every number the v2.1 cards and charts use.

Usage: python3 scripts/charts/data_v21.py
Common basis: v2's 277-prompt sealed harmful test minus its 14 cbrn_weapons prompts = 263 prompts, the
set v2.1 was evaluated on. Every model is recomputed per row on exactly those ids.
Only id, category, refusal, cap_hit and judge numeric fields are read; text fields are never loaded.

Sources (all llama.cpp 19bc41dbb, greedy, think cap 4,096, answer cap 8,192, Jev jev-1.14.0):
  ~/.inkling-abl/v11b_private/val/out/test8k_*/   original, v1.0, v2 and the public releases (v2 campaign)
  ~/.inkling-abl/v3_private/ev/v31C/old_*          v2.1 (= v3.1 candidate C)
  ~/.inkling-abl/v3_private/ev/v2/old_*            v2 re-run alongside v2.1 (consistency check only)
  ~/.inkling-abl/v3_private/oldtest/bison_*        independent second judge (Bison) for some baselines
"""
from __future__ import annotations

import collections
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v3"))
import agg_v3  # noqa: E402  (canonical paired_diff used for the v2 -> v2.1 history numbers)

R = os.path.abspath(os.path.join(HERE, "..", "..")) + "/"
H = os.path.expanduser("~/.inkling-abl/")
V = H + "v11b_private/val/out/"
EV = H + "v3_private/ev/"
OLD = H + "v3_private/oldtest/"
V2DATA = json.load(open(R + "package_v2/data_test8k.json"))
EXCLUDED = "cbrn_weapons"
MODES = ("off", "e09")
N_BOOT = 10000

CAMPAIGN = {"orig": "test8k_orig", "v1": "test8k_v10", "v2": "test8k_E", "atomic_lora": "test8k_atomic",
            "atomic_adq4": "test8k_adq4", "xq": "test8k_xq", "ou": "test8k_ou", "heretic": "test8k_heretic"}
BISON = {"orig": "orig", "v1": "v10", "v2": "E", "atomic_lora": "atomic"}
G_KEYS, J_KEYS = ("id", "category", "refusal", "cap_hit"), ("id", "useful", "held_back", "error")


# ---- loading (numeric / id fields only) ------------------------------------
def rows(path: str, keep: tuple[str, ...]) -> dict[str, dict]:
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["id"]] = {k: r.get(k) for k in keep}
    return out


def judged(path: str) -> dict[str, int]:
    """id -> useful (0/1); rows with a judge error are left out."""
    return {i: int(r["useful"]) for i, r in rows(path, J_KEYS).items() if not r["error"] and r["useful"] is not None}


def load_model(gens_tpl: str, jev_tpl: str) -> dict[str, dict]:
    return {m: {"gens": rows(gens_tpl.format(m=m), G_KEYS), "jev": judged(jev_tpl.format(m=m))} for m in MODES}


def load_all() -> dict[str, dict]:
    d = {k: load_model(V + c + "/gens_harmful_{m}.jsonl", V + c + "/jev_harmful_{m}.jsonl") for k, c in CAMPAIGN.items()}
    d["v21"] = load_model(EV + "v31C/old_{m}.jsonl", EV + "v31C/jev_old_{m}.jsonl")
    d["v2_rerun"] = load_model(EV + "v2/old_{m}.jsonl", EV + "v2/jev_old_{m}.jsonl")
    for k, b in BISON.items():
        for m in MODES:
            d[k][m]["bison"] = judged(f"{OLD}bison_{b}_{m}.jsonl")
    for m in MODES:
        d["v21"][m]["bison"] = judged(f"{EV}v31C/bison_old_{m}.jsonl")
    return d


# ---- statistics -------------------------------------------------------------
def boot(vals: list[float], seed: int) -> list[float]:
    a = np.asarray(vals, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(N_BOOT, len(a)))
    lo, hi = np.percentile(a[idx].mean(axis=1), [2.5, 97.5])
    return [round(100 * lo, 1), round(100 * hi, 1)]


def pct(num: float, den: int) -> float:
    return round(100 * num / den, 1)


def rate(vals: list[int], seed: int = 1234) -> dict:
    return {"value": pct(sum(vals), len(vals)), "ci": boot(vals, seed), "count": sum(vals), "n": len(vals)}


def summary(d: dict, basis: list[str]) -> dict:
    out: dict = {"n": len(basis)}
    for m in MODES:
        g, j = d[m]["gens"], d[m]["jev"]
        u = [j[i] for i in basis]
        out[f"useful_{m}"] = rate(u)
        out[f"refusal_{m}"] = {"value": pct(sum(g[i]["refusal"] for i in basis), len(basis)),
                               "count": sum(g[i]["refusal"] for i in basis), "n": len(basis)}
        out[f"thinkcap_{m}"] = {"value": pct(sum(bool(g[i]["cap_hit"]) for i in basis), len(basis)),
                                "count": sum(bool(g[i]["cap_hit"]) for i in basis), "n": len(basis)}
        cat = collections.defaultdict(list)
        for i in basis:
            cat[g[i]["category"]].append(j[i])
        out[f"useful_{m}_by_cat"] = {c: pct(sum(v), len(v)) for c, v in sorted(cat.items())}
        out["n_by_cat"] = {c: len(v) for c, v in sorted(cat.items())}
        if "bison" in d[m]:
            b = d[m]["bison"]
            out[f"bison_useful_{m}"] = rate([b[i] for i in basis if i in b])
    return out


def head_to_head(a: dict, b: dict, basis: list[str], seed: int = 7) -> dict:
    """a vs b on the same prompts: counts plus paired bootstrap CI of the difference in useful rate."""
    d = [a[i] - b[i] for i in basis]
    return {"a_only": d.count(1), "b_only": d.count(-1),
            "both": sum(1 for i in basis if a[i] and b[i]), "neither": sum(1 for i in basis if not a[i] and not b[i]),
            "diff": pct(sum(d), len(d)), "ci": boot(d, seed), "n": len(d)}


# ---- v2 -> v2.1 history and decoding (unchanged definitions from the earlier v2.1 build) -------
def paired_judge(f: str, judge: str, metric: str, a: str = "v2", b: str = "v31C") -> dict:
    res = agg_v3.paired_diff(agg_v3.load(f"{EV}{a}/{f}.jsonl"), agg_v3.load(f"{EV}{a}/{judge}_{f}.jsonl"),
                             agg_v3.load(f"{EV}{b}/{f}.jsonl"), agg_v3.load(f"{EV}{b}/{judge}_{f}.jsonl"), metric)
    x = list(res.values())[0]
    lo, hi = x["b_minus_a"]["CI"]
    r2 = lambda v: round(v * 100, 2)  # noqa: E731
    return {"v2": r2(x["a"]["value"]), "v21": r2(x["b"]["value"]), "delta": r2(x["b_minus_a"]["value"]),
            "ci": [r2(lo), r2(hi)], "n": x["n_paired"], "significant": lo > 0 or hi < 0}


def cap_map(path: str) -> dict:
    return {(r["id"], r.get("turn", 0)): bool(r.get("cap_hit")) for r in agg_v3.load(path)}


def cap_paired(ra: dict, rb: dict) -> dict:
    """Replicates scripts/v3/paired.cap_diff (seed 0, 2,000 resamples)."""
    keys = sorted(set(ra) & set(rb))
    d = [int(rb[k]) - int(ra[k]) for k in keys]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    pa, pb = sum(ra[k] for k in keys) / len(keys), sum(rb[k] for k in keys) / len(keys)
    lo, hi = boots[50], boots[1949]
    return {"a": round(pa * 100, 2), "b": round(pb * 100, 2), "delta": round((pb - pa) * 100, 2),
            "ci": [round(lo * 100, 2), round(hi * 100, 2)], "n": len(keys), "significant": lo > 0 or hi < 0}


def settings_data() -> dict:
    runs = {"v2_greedy": EV + "v2/test_e09.jsonl", "v21_greedy": EV + "v31C/test_e09.jsonl",
            "v2_t06": EV + "decode/e09_t06.jsonl", "v21_t06": EV + "v31C_t06/test_e09.jsonl"}
    maps = {k: cap_map(p) for k, p in runs.items()}
    common = sorted(set.intersection(*(set(m) for m in maps.values())))
    rates = {}
    for k, mp in maps.items():
        vals = [float(mp[x]) for x in common]
        lo, hi = agg_v3.boot_ci(vals)
        rates[k] = {"value": round(100 * sum(vals) / len(vals), 2), "ci": [round(lo * 100, 2), round(hi * 100, 2)],
                    "n": len(vals)}
    pairs = {"v21_t06_minus_v21_greedy": ("v21_greedy", "v21_t06"), "v21_t06_minus_v2_greedy": ("v2_greedy", "v21_t06"),
             "v2_t06_minus_v2_greedy": ("v2_greedy", "v2_t06"), "v21_greedy_minus_v2_greedy": ("v2_greedy", "v21_greedy")}
    return {"set": "v2.1's own sealed test (new prompts), harmful, thinking on (effort 0.9)", "common_n": len(common),
            "think_cap_rates": rates, "paired": {n: cap_paired(maps[a], maps[b]) for n, (a, b) in pairs.items()},
            "bison_useful_v21_t06_vs_greedy": paired_judge("test_e09", "bison", "useful", "v31C", "v31C_t06")}


def history_data() -> dict:
    sets = ("test_off", "test_e09", "old_off", "old_e09")
    return {"_note": "v2.1 minus v2 on the same prompts, paired bootstrap (agg_v3.paired_diff). test = v2.1's own "
                     "sealed test (480 prompts off, 240 on); old = the 263-prompt basis.",
            "useful": {f: {j: paired_judge(f, j, "useful") for j in ("jev", "bison")} for f in sets},
            "held_back": {f: {j: paired_judge(f, j, "held_back") for j in ("jev", "bison")} for f in sets}}


def capability_v21() -> dict:
    cap = json.load(open(EV + "v31C/cap/metrics.json"))["metrics"]
    b = lambda k: {"value": round(cap[k]["value"], 1), "ci": [round(x, 1) for x in cap[k]["CI"]], "n": cap[k]["n"]}  # noqa: E731
    hits = n = 0
    for s in range(8):
        m = json.load(open(f"{EV}v31C/bl_s{s}/metrics.json"))["metrics"]["overrefusal_off"]
        hits, n = hits + round(m["value"] * m["n"]), n + m["n"]
    return {"mmlu300": b("mmlu300"), "gsm8k100": b("gsm8k100"), "overrefusal": {"value": pct(hits, n), "count": hits, "n": n}}


# ---- main -------------------------------------------------------------------
def main() -> None:
    d = load_all()
    basis = sorted(d["v21"]["off"]["gens"])
    full = set(d["v2"]["off"]["gens"])
    assert set(d["v21"]["e09"]["gens"]) == set(basis) and set(basis) <= full
    assert {d["v2"]["off"]["gens"][i]["category"] for i in full - set(basis)} == {EXCLUDED}
    assert all(EXCLUDED != d["v21"]["off"]["gens"][i]["category"] for i in basis)
    for k in d:
        for m in MODES:
            assert all(i in d[k][m]["jev"] for i in basis), (k, m)

    models = {k: summary(d[k], basis) for k in ["v21", *CAMPAIGN]}
    h2h = {f"v21_vs_{o}_{m}": head_to_head(d["v21"][m]["jev"], d[o][m]["jev"], basis)
           for o in CAMPAIGN for m in MODES}
    rerun = summary(d["v2_rerun"], basis)
    static = {k: dict(v) for k, v in V2DATA["static"].items()}
    cap = capability_v21()
    static["v21"] = {"label": "Unbound v2.1 (ours)", "precision": "Q8_0", "kld": 0.0439, "top1": 95.4,
                     "mmlu300": cap["mmlu300"]["value"], "mmlu300_ci": cap["mmlu300"]["ci"],
                     "gsm8k100": cap["gsm8k100"]["value"], "overrefusal": cap["overrefusal"]["value"],
                     "overrefusal_count": cap["overrefusal"]["count"], "overrefusal_n": cap["overrefusal"]["n"]}
    out = {
        "_meta": {"basis": "v2's sealed harmful test (277) minus its 14 cbrn_weapons prompts = 263 prompts; "
                           "the CBRN category is excluded for every model",
                  "n": len(basis), "excluded_category": EXCLUDED, "engine": "llama.cpp 19bc41dbb, greedy",
                  "caps": {"think": 4096, "answer": 8192}, "judge": "Jev jev-1.14.0, useful = p_fulfils >= 0.5",
                  "second_judge": "Bison (independent, never used for training selection), where judged",
                  "refusal": "phrase-list flag on the first 300 characters of the answer (same flag for all)",
                  "ci": "95% bootstrap, 10,000 resamples (numpy, seed 1234; paired head-to-head seed 7)",
                  "v2_source": "the v2 release campaign run (test8k_E), so every competitor row comes from the same "
                               "campaign; v2.1 from its own eval run on the identical engine and settings",
                  "date": "2026-11-24"},
        "models": models, "head_to_head": h2h, "static": static, "releases": V2DATA["releases"],
        "v2_rerun_check": {"_note": "v2 re-run in the v2.1 eval batch vs the v2 release campaign, same 263 ids",
                           "rerun": {m: rerun[f"useful_{m}"] for m in MODES},
                           "campaign": {m: models["v2"][f"useful_{m}"] for m in MODES},
                           "gap_pts": {m: round(rerun[f"useful_{m}"]["value"] - models["v2"][f"useful_{m}"]["value"], 1)
                                       for m in MODES},
                           "paired_campaign_vs_rerun": {m: head_to_head(d["v2"][m]["jev"], d["v2_rerun"][m]["jev"], basis)
                                                        for m in MODES}},
        "capability_v21": cap, "settings": settings_data(), "history": history_data(),
    }
    path = R + "package_v21/data_v21.json"
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
    write_values(out)



# ---- values.json for the card templates -------------------------------------
KEEP_PREFIX = ("CHART_REV", "SIZE_", "SHA_", "KLD_Q", "KLD_IQ", "TOP1_", "BASALT_", "LORA_", "FT_RANK", "CKPT_SHA")
SHORT = {"orig": "ORIG", "v1": "V1", "v2": "V2", "v21": "V21", "atomic_lora": "ATOM", "atomic_adq4": "ADQ4",
         "xq": "XQ", "ou": "OU", "heretic": "HER"}


def f1(v: float) -> str:
    return f"{v:.1f}%"


def sci(c: list[float]) -> str:
    return f"[{c[0]:+.1f}, {c[1]:+.1f}]"


def model_values(v: dict, data: dict) -> None:
    for k, s in SHORT.items():
        x = data["models"][k]
        for mode, tag in (("off", "OFF"), ("e09", "E09")):
            v[f"{s}_{tag}"] = f1(x[f"useful_{mode}"]["value"])
            v[f"{s}_{tag}_CI"] = f"[{x[f'useful_{mode}']['ci'][0]:.1f}, {x[f'useful_{mode}']['ci'][1]:.1f}]"
            v[f"{s}_CAP_{tag}"] = f1(x[f"thinkcap_{mode}"]["value"])
            if f"bison_useful_{mode}" in x:
                b = x[f"bison_useful_{mode}"]
                v[f"{s}_BISON_{tag}"] = f1(b["value"])
                v[f"{s}_BISON_{tag}_N"] = str(b["n"])
        v[f"{s}_REF"] = f"{f1(x['refusal_off']['value'])} / {f1(x['refusal_e09']['value'])}"
        if k != "v21":
            for mode, tag in (("off", "OFF"), ("e09", "E09")):
                h = data["head_to_head"][f"v21_vs_{k}_{mode}"]
                v[f"H2H_{s}_{tag}_D"] = f"{h['diff']:+.1f}"
                v[f"H2H_{s}_{tag}_CI"] = sci(h["ci"])
                v[f"H2H_{s}_{tag}_WIN"] = str(h["a_only"])
                v[f"H2H_{s}_{tag}_LOSS"] = str(h["b_only"])


def history_values(v: dict, data: dict) -> None:
    hist = data["history"]
    for f, tag in (("test_off", "NEWOFF"), ("test_e09", "NEWON"), ("old_off", "OLDOFF"), ("old_e09", "OLDON")):
        for j in ("jev", "bison"):
            for blk, pre in (("useful", "U"), ("held_back", "H")):
                d = hist[blk][f][j]
                key = f"{pre}_{tag}_{j.upper()}"
                v[f"{key}_V2"], v[f"{key}_V21"] = f"{d['v2']:.1f}", f"{d['v21']:.1f}"
                v[f"{key}_D"], v[f"{key}_CI"], v[f"{key}_N"] = f"{d['delta']:+.1f}", sci(d["ci"]), str(d["n"])
    s = data["settings"]
    r, p, b = s["think_cap_rates"], s["paired"], s["bison_useful_v21_t06_vs_greedy"]
    v["LOOP_N"] = str(s["common_n"])
    for k in ("v2_greedy", "v21_greedy", "v2_t06", "v21_t06"):
        v[f"LOOP_{k.upper()}"] = f1(r[k]["value"])
    for name, key in (("v21_t06_minus_v21_greedy", "LOOP_T06_VS_GREEDY"), ("v21_t06_minus_v2_greedy", "LOOP_T06_VS_V2"),
                      ("v2_t06_minus_v2_greedy", "LOOP_V2T06_VS_V2")):
        v[f"{key}_D"], v[f"{key}_CI"] = f"{p[name]['delta']:+.1f}", sci(p[name]["ci"])
    v["T06_BISON_GREEDY"], v["T06_BISON_T06"] = f"{b['v2']:.1f}", f"{b['v21']:.1f}"
    v["T06_BISON_D"], v["T06_BISON_CI"] = f"{b['delta']:+.1f}", sci(b["ci"])


def write_values(data: dict) -> None:
    path = R + "package_v21/values.json"
    old = json.load(open(path)) if os.path.exists(path) else {}
    v = {k: x for k, x in old.items() if k.startswith(KEEP_PREFIX)}
    model_values(v, data)
    history_values(v, data)
    st = data["static"]
    v21, cap = st["v21"], data["capability_v21"]
    v["N_BASIS"] = str(data["_meta"]["n"])
    v["KLD_V21"], v["TOP1_V21"] = f"{v21['kld']:.4f}", f"{v21['top1']:.1f}%"
    v["MMLU_V21"], v["MMLU_V21_CI"] = f"{v21['mmlu300']:.1f}", f"[{v21['mmlu300_ci'][0]:.1f}, {v21['mmlu300_ci'][1]:.1f}]"
    v["GSM_V21"] = f"{v21['gsm8k100']:.0f}"
    v["OVERREF_V21"] = f"{cap['overrefusal']['count']} of {cap['overrefusal']['n']} ({cap['overrefusal']['value']:.1f}%)"
    v["OVERREF_V21_PCT"] = f1(cap["overrefusal"]["value"])
    rc = data["v2_rerun_check"]
    v["V2_RERUN_OFF"], v["V2_RERUN_E09"] = f1(rc["rerun"]["off"]["value"]), f1(rc["rerun"]["e09"]["value"])
    v["V2_RERUN_GAP_OFF"], v["V2_RERUN_GAP_E09"] = f"{rc['gap_pts']['off']:+.1f}", f"{rc['gap_pts']['e09']:+.1f}"
    v["V21_REF_OFF_N"] = str(data["models"]["v21"]["refusal_off"]["count"])
    v["V21_REF_E09_N"] = str(data["models"]["v21"]["refusal_e09"]["count"])
    json.dump(v, open(path, "w"), indent=1)
    print("wrote", path, len(v), "keys")


if __name__ == "__main__":
    main()
