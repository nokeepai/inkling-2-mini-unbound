"""Auditor (Mac): robustness summary per model from the private dir (all *_tr / *_r? eval dirs).
Per set: phrase refusal on the raw answer, phrase refusal on raw OR back-translation (lang_*), Jev useful
(from jev_metrics.json written by jev_sealed.py), thinking-cap hits and loops. Writes R/audit/test/robust.json.

  python3 robust_table.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/Users/nikshepsvn/Code/inkling-abl/harness")
import judge_jev as J  # noqa: E402  (boot only)

P = "/Users/nikshepsvn/.inkling-abl/audit_private/test"
OUT = "/Users/nikshepsvn/Code/inkling-abl/audit/test/robust.json"


def st(xs):
    return {"value": sum(xs) / len(xs), "CI": J.boot(xs), "n": len(xs)} if xs else None


res: dict = {}
for c in ("ou", "c1h_r1_svd1_s3_L4", "c1dt001"):
    jev: dict = {}
    for jm in glob.glob(f"{P}/{c}_r*/jev_metrics.json") + glob.glob(f"{P}/{c}_tr/jev_metrics.json"):
        jev.update(json.load(open(jm))["metrics"])
    for f in sorted(glob.glob(f"{P}/{c}_r*/gens_robust_*_off.jsonl") + glob.glob(f"{P}/{c}_tr/gens_robust_*_off.jsonl")):
        if f.endswith(".en.jsonl"):
            continue
        name = os.path.basename(f)[5:-6]
        rows = [json.loads(l) for l in open(f)]
        en = f[:-6] + ".en.jsonl"
        d = {"phrase_raw": st([float(r["refusal"]) for r in rows]),
             "cap_hit": st([float(r["cap_hit"]) for r in rows]), "loop": st([float(r.get("loop", 0)) for r in rows]),
             "jev_useful": jev.get(f"jev_useful_{name}"),
             "phrase_raw_or_backtranslated": jev.get(f"robust_refusal_phrase_en_{name}")}
        res.setdefault(c, {})[name] = d
json.dump(res, open(OUT, "w"), indent=1)
for c, sets in res.items():
    for name, d in sets.items():
        f = lambda m: f"{100 * m['value']:.1f}" if m else "–"  # noqa: E731
        print(f"{c:20s} {name:24s} phrase {f(d['phrase_raw']):>5}  phrase+bt {f(d['phrase_raw_or_backtranslated']):>5}  "
              f"jev useful {f(d['jev_useful']):>5}  cap {f(d['cap_hit']):>4} loop {f(d['loop']):>4}")
