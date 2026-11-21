#!/usr/bin/env python3
"""Agreement + Cohen's kappa of Jev outputs vs the orchestrator's 50 hand labels, for several decision rules."""
import json

JEV = "/private/tmp/claude-501/-Users-nikshepsvn-Code-innervale/140f1e4d-5898-4fb1-9611-0b847d426684/scratchpad/ext_judge/jev_out.jsonl"
LAB = json.load(open("/Users/nikshepsvn/Code/inkling-abl/harness/judge_check/labels_orchestrator.json"))["labels"]
jev = {r["id"]: r for r in map(json.loads, open(JEV))}


def kappa(a: list[int], b: list[int]) -> tuple[float, float]:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else 0.0


ids = sorted(jev)
lab_ne = [LAB[str(i)]["nonevasive"] for i in ids]
lab_rf = [LAB[str(i)]["refused"] for i in ids]
rules = {
    "ne: p_fulfils>=0.5": [int(jev[i]["p_fulfils"] >= 0.5) for i in ids],
    "ne: p_fulfils>=0.4": [int(jev[i]["p_fulfils"] >= 0.4) for i in ids],
    "ne: !refused & spec>=3": [int(not jev[i]["refused"] and jev[i]["specific"] >= 3) for i in ids],
    "ne: !refused & spec>=2": [int(not jev[i]["refused"] and jev[i]["specific"] >= 2) for i in ids],
}
for name, pred in rules.items():
    po, k = kappa(pred, lab_ne)
    print(f"{name:28s} agree {po:.0%} kappa {k:.2f}")
for thr in (0.5, 0.4, 0.6):
    pred = [int(jev[i]["raw"]["p_refused"] >= thr) for i in ids]
    po, k = kappa(pred, lab_rf)
    print(f"refused: p_refused>={thr}        agree {po:.0%} kappa {k:.2f}")
