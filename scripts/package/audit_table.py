"""Print the auditor's sealed-test aggregate metrics (orig / ou / winner) side by side, plus the worst category.
Reads only R/audit/test/evals/*/metrics.json and jev_metrics.json (aggregates; no prompts)."""
import json, os, sys
R = os.path.expanduser("~/Code/inkling-abl/audit/test/evals")
W = sys.argv[1] if len(sys.argv) > 1 else "c1h_r1_svd1_s3_L4"
M = ["orig", "ou", W]


def load(c, tag, f="metrics.json"):
    p = f"{R}/{c}_{tag}/{f}"
    return json.load(open(p))["metrics"] if os.path.exists(p) else {}


def fmt(m):
    if not m:
        return "-"
    v, ci = m["value"], m.get("CI")
    s = f"{v:.4f}" if v < 1.001 else f"{v:.1f}"
    return s + (f" [{ci[0]:.4f},{ci[1]:.4f}]" if ci else "") + f" n={m.get('n')}"


for tag, f in (("th", "metrics.json"), ("th", "jev_metrics.json"), ("tb", "metrics.json")):
    d = {c: load(c, tag, f) for c in M}
    keys = sorted(set().union(*[set(x) for x in d.values()]))
    print(f"== {tag} {f}")
    for k in keys:
        if "_cat_" in k:
            continue
        print(k, " | ".join(fmt(d[c].get(k)) for c in M))
    for c in M:
        cats = {k: v for k, v in d[c].items() if "_cat_" in k and k.startswith(("refusal_", "jev_useful"))}
        for pre in sorted({k.split("_cat_")[0] for k in cats}):
            sub = {k: v for k, v in cats.items() if k.startswith(pre + "_cat_")}
            worst = (max if pre.startswith("refusal") else min)(sub, key=lambda k: sub[k]["value"])
            print(f"  {c} worst {pre}: {worst.split('_cat_')[1]} {sub[worst]['value']:.4f}")
