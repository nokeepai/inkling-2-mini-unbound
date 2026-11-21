"""Collect all mpoa candidates' fleet metrics -> methods/mpoa/all_metrics.tsv and a Pareto table (markdown to stdout)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import metrics as MET

R = os.path.expanduser("~/Code/inkling-abl")
allm = MET.collect(["c1", "c2"], exclude_pods=("ink-eval-2",))
base = MET.collect(["orig", "ou", "c0"], exclude_pods=("ink-eval-2",))
COLS = ["kl", "top1", "screen_refusal_off", "screen_mmlu300", "refusal_off", "refusal_e09", "refusal_combined_off",
        "refusal_combined_e09", "mmlu300", "gsm8k100", "harmful_e09_cap_hit", "harmful_e09_rep", "probe_honest_limit_off",
        "probe_honest_limit_e09", "probe_crisis_resources_off", "overrefusal200_off", "judge_nonevasive_e09",
        "judge_nonevasive_off"]


def spec(c):
    p = f"{R}/methods/mpoa/cands/{c}/spec.json"
    if not os.path.exists(p):
        return {}
    s = json.load(open(p))
    pr = dict(s["params"])
    pr.pop("s_list", None)
    pr.pop("harmless_proj", None)
    return {"params": pr, "s": s["diag"]["s"]}


with open(f"{R}/methods/mpoa/all_metrics.tsv", "w") as f:
    f.write("cand\t" + "\t".join(COLS) + "\tparams\n")
    for c, m in sorted(allm.items()):
        f.write(c + "\t" + "\t".join(str(m.get(k, "")) for k in COLS) + "\t" + json.dumps(spec(c)) + "\n")


def ref_of(m):
    if "refusal_off" in m and "refusal_e09" in m:
        return (m["refusal_off"] + m["refusal_e09"]) / 2, "full"
    if "screen_refusal_off" in m:
        return m["screen_refusal_off"], "screen"
    return None, None


pts = []
for c, m in allm.items():
    r, st = ref_of(m)
    if r is None or "kl" not in m:
        continue
    pts.append((c, r, m["kl"], st, m))
front = [p for p in pts if not any(q[1] <= p[1] and q[2] <= p[2] and (q[1] < p[1] or q[2] < p[2]) for q in pts)]
front.sort(key=lambda p: p[1])
fmt = lambda v, d=1, pct=False: "" if v is None else (f"{100 * v:.{d}f}" if pct else f"{v:.{d}f}")
print("| cand | stage | refusal % (mean off/e0.9 or screen off) | KL | top1 % | MMLU | GSM8K | e0.9 cap % | e0.9 rep % | honest off/e09 | over-ref % | judge non-evasive e0.9 % | params |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for c, r, kl, st, m in front:
    g = m.get
    print(f"| {c} | {st} | {fmt(r, 1, True)} | {kl:.4f} | {fmt(g('top1'), 1, True)} | {fmt(g('mmlu300', g('screen_mmlu300')))} | "
          f"{fmt(g('gsm8k100'), 0)} | {fmt(g('harmful_e09_cap_hit'), 1, True)} | {fmt(g('harmful_e09_rep'), 1, True)} | "
          f"{fmt(g('probe_honest_limit_off'), 2)}/{fmt(g('probe_honest_limit_e09'), 2)} | {fmt(g('overrefusal200_off'), 1, True)} | "
          f"{fmt(g('judge_nonevasive_e09'), 1, True)} | `{json.dumps(spec(c).get('params', {}), separators=(',', ':'))[:160]}` |")
print()
print("baselines:", json.dumps({b: {k: round(v, 4) for k, v in base.get(b, {}).items() if k in ('kl', 'refusal_off', 'refusal_e09', 'mmlu300', 'gsm8k100', 'harmful_e09_cap_hit', 'harmful_e09_rep', 'overrefusal200_off')} for b in ("orig", "ou")}))
print("n candidates with KL:", len(pts), "full:", sum(1 for p in pts if p[3] == "full"))
