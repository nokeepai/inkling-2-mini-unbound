#!/usr/bin/env python3
"""Leaderboard from results/runs.jsonl: latest value per (candidate, metric) for the key metrics.
Fleet rows only for refusal/capability; kl_worker rows for KL. Prints candidates sorted by KL."""
import json, collections

ROWS = [json.loads(l) for l in open("/Users/nikshepsvn/Code/inkling-abl/results/runs.jsonl") if l.strip()]
KEYS = {
    "kl": ("kl", "kl_256"), "top1": ("top1", "top1_256"),
    "scr_ref": ("screen_refusal_off",), "ref_off": ("refusal_off", "harmful_refusal_off"),
    "ref_e09": ("refusal_e09", "harmful_refusal_e09"), "mmlu": ("mmlu300", "screen_mmlu300"),
    "gsm": ("gsm8k100",), "over": ("overrefusal200_off",), "hon": ("probe_honest_limit",),
    "cap": ("harmful_e09_cap_hit",), "rep": ("harmful_e09_rep",),
    "hb_use_off": ("judge_nonevasive_off", "screen_judge_nonevasive_off"),
}
best = collections.defaultdict(dict)
for r in ROWS:
    cand, m = r.get("candidate"), r.get("metric")
    agent = str(r.get("agent", ""))
    for k, names in KEYS.items():
        if m in names:
            if k in ("kl", "top1") and r.get("engine") != "hf_w4a16":
                continue
            if k not in ("kl", "top1") and not ("fleet-2" in agent or "judge_relay" in agent):
                continue
            if k.startswith("hb") and "harmbench" not in str(r.get("scorer", "")):
                continue
            best[cand][k] = r.get("value")
cols = ["kl", "top1", "scr_ref", "ref_off", "ref_e09", "mmlu", "gsm", "over", "hon", "cap", "rep", "hb_use_off"]
print("cand".ljust(12) + "".join(c.rjust(10) for c in cols))
def fmt(v):
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) and v < 1 else f"{v:.2f}" if isinstance(v, float) else str(v))
for cand, d in sorted(best.items(), key=lambda kv: (kv[1].get("kl") is None, kv[1].get("kl") or 9)):
    if any(k in d for k in cols[2:]) or (d.get("kl") or 9) < 0.03:
        print(str(cand)[:12].ljust(12) + "".join(fmt(d.get(c)).rjust(10) for c in cols))
