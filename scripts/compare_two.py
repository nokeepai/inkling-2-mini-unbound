#!/usr/bin/env python3
"""Side-by-side val comparison of candidates from runs.jsonl (fleet-2 + kl_worker + judge rows),
plus honest-limit probe failures printed in full for hand inspection."""
import json, glob, sys

R = "/Users/nikshepsvn/Code/inkling-abl"
CANDS = sys.argv[1:] or ["c1h_svd1_s2", "c1t013", "ou", "orig"]
rows = [json.loads(l) for l in open(f"{R}/results/runs.jsonl") if l.strip()]
WANT = ["refusal_off", "refusal_e09", "refusal_combined_off", "refusal_combined_e09", "judge_nonevasive_off",
        "judge_nonevasive_e09", "kl_256", "top1_256", "probe_honest_limit_off", "probe_honest_limit_e09",
        "probe_crisis_resources_off", "probe_crisis_resources_e09", "overrefusal200_off", "harmful_e09_cap_hit",
        "harmful_e09_rep", "borderline200_loop", "gsm8k_loop", "harmful_e09_loop", "mmlu300", "gsm8k100"]
val = {c: {} for c in CANDS}
for r in rows:
    c, m = r.get("candidate"), r.get("metric")
    if c not in val or m not in WANT:
        continue
    a = str(r.get("agent", ""))
    if m.startswith(("kl", "top1")) and r.get("engine") != "hf_w4a16":
        continue
    if not m.startswith(("kl", "top1")) and not ("fleet-2" in a or "judge_relay" in a):
        continue
    if "judge" in m or "combined" in m:
        if "harmbench" not in str(r.get("scorer", "")):
            continue
    val[c][m] = r.get("value")
print("metric".ljust(28) + "".join(c[:14].rjust(15) for c in CANDS))
for m in WANT:
    print(m.ljust(28) + "".join((f"{val[c][m]:.4f}" if isinstance(val[c].get(m), float) else str(val[c].get(m, "—"))).rjust(15) for c in CANDS))
print("\nHONEST-LIMIT PROBES (category honest/limit) per candidate:")
for c in CANDS:
    d = sorted(glob.glob(f"{R}/results/outputs/{c}/full-*"))
    if not d:
        continue
    for mode in ("off", "e09"):
        ps = [json.loads(l) for l in open(f"{d[-1]}/gens_probes_{mode}.jsonl")]
        hl = [p for p in ps if p.get("category") != "crisis"]
        print(f"== {c} {mode}: n={len(hl)} categories={sorted(set(p.get('category') for p in ps))}")
        for p in hl:
            print(f"   [{p['id'][-6:]}] Q: {p['prompt'][:80]!r}\n        A: {p['answer'][:220]!r}")
