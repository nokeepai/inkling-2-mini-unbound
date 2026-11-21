"""Write R/harness/borderline200_ids.json once: a seeded (1234), category-stratified 200-prompt subset of borderline val
(same method as screen_ids.json). Refuses to overwrite."""
import json
import os
import random

R = os.path.expanduser("~/Code/inkling-abl")
OUT = f"{R}/harness/borderline200_ids.json"
N, SEED = 200, 1234

rows = [json.loads(l) for l in open(f"{R}/data/val/borderline.jsonl")]
by: dict[str, list[str]] = {}
for r in rows:
    by.setdefault(r["category"], []).append(r["id"])
quota = {c: N * len(v) / len(rows) for c, v in by.items()}
alloc = {c: int(q) for c, q in quota.items()}
for c in sorted(quota, key=lambda c: (-(quota[c] - alloc[c]), c))[: N - sum(alloc.values())]:
    alloc[c] += 1
rng = random.Random(SEED)
ids: list[str] = []
for c in sorted(by):
    ids += rng.sample(sorted(by[c]), alloc[c])
assert len(ids) == N == len(set(ids))
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} exists; not overwriting")
json.dump({"n": N, "seed": SEED, "source": "data/val/borderline.jsonl",
           "method": "proportional stratified by category, largest remainder",
           "alloc": dict(sorted(alloc.items())), "ids": sorted(ids)}, open(OUT, "w"), indent=1)
print(alloc)
