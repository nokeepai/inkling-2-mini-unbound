"""Hand-check samples for the hygiene redefinition (owner): from the think-cap-4096 reruns in results/outputs/*/full*C4096*/,
20 random rows the NEW detector flags (loop == 1) and 20 random rows the OLD one flags (rep8x4 == 1), any models and
sets, seed 1234. Writes R/audit/hygiene_samples/{new_loop_flagged.jsonl, old_rep8x4_flagged.jsonl, summary.json};
each row {model, set, id, think (full), answer, loop, rep8x4, think_len, cap_hit}.
  python3 hygiene_samples.py"""
import glob
import json
import os
import random

R = os.path.expanduser("~/Code/inkling-abl")
OUT = f"{R}/audit/hygiene_samples"
SETS = ("harmful_e09", "harmful_off", "borderline200", "gsm8k")
rows = []
counts: dict[str, dict] = {}
for d in sorted(glob.glob(f"{R}/results/outputs/*/full*C4096*/")):
    if not os.path.exists(f"{d}/fleet_done.json"):
        continue
    model = d.rstrip("/").split("/")[-2]
    for s in SETS:
        p = f"{d}/gens_{s}.jsonl"
        if not os.path.exists(p):
            continue
        for l in open(p):
            r = json.loads(l)
            row = {"model": model, "set": s, "id": r.get("id"), "think": r.get("think", ""), "answer": r.get("answer", ""),
                   "loop": r.get("loop"), "rep8x4": r.get("rep8x4"), "think_len": r.get("think_len"),
                   "cap_hit": r.get("cap_hit")}
            rows.append(row)
            c = counts.setdefault(f"{model}/{s}", {"n": 0, "loop": 0, "rep8x4": 0, "both": 0, "cap_hit": 0})
            c["n"] += 1
            c["loop"] += int(bool(row["loop"]))
            c["rep8x4"] += int(bool(row["rep8x4"]))
            c["both"] += int(bool(row["loop"]) and bool(row["rep8x4"]))
            c["cap_hit"] += int(bool(row["cap_hit"]))
os.makedirs(OUT, exist_ok=True)
for name, key in (("new_loop_flagged", "loop"), ("old_rep8x4_flagged", "rep8x4")):
    flagged = sorted((r for r in rows if r[key]), key=lambda r: (r["model"], r["set"], str(r["id"])))
    pick = random.Random(1234).sample(flagged, min(20, len(flagged)))
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in pick:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(flagged)} flagged, wrote {len(pick)}")
json.dump({"runs": sorted({r['model'] for r in rows}), "per_model_set": counts, "seed": 1234,
           "new_detector": "span >= 32 tokens repeated >= 3 times back to back in think+answer",
           "old_detector": "any token 8-gram occurring >= 4 times in think+answer"},
          open(f"{OUT}/summary.json", "w"), indent=1)
