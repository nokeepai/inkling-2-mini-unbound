"""Build R/harness/judge_check/check_in.jsonl: 25 orig + 25 OU harmful-val responses (thinking stripped).
Sampling: seed 1234; mode alternates off/e09; OU items stratified so roughly half are phrase-refusals if available.
  python3 build_judge_check.py <orig_eval_dir> <ou_eval_dir>
"""
import json
import os
import random
import sys

R = os.path.expanduser("~/Code/inkling-abl")
sys.path.insert(0, f"{R}/harness")
from sgl_rescore import is_refusal  # noqa: E402

rng = random.Random(1234)
out = []
for tag, d in (("orig", sys.argv[1]), ("ou", sys.argv[2])):
    pool = []
    for mode in ("off", "e09"):
        for l in open(f"{d}/gens_harmful_{mode}.jsonl"):
            r = json.loads(l)
            pool.append({"model": tag, "mode": mode, "id": r["id"], "category": r["category"], "prompt": r["prompt"],
                         "response": r["answer"], "phrase_refusal": int(is_refusal(r["answer"]))})
    ref = [p for p in pool if p["phrase_refusal"]]
    non = [p for p in pool if not p["phrase_refusal"]]
    rng.shuffle(ref)
    rng.shuffle(non)
    k_non = min(len(non), 13)
    pick = non[:k_non] + ref[:25 - k_non]
    if len(pick) < 25:
        pick += non[k_non:k_non + 25 - len(pick)]
    out += pick[:25]
rng.shuffle(out)
os.makedirs(f"{R}/harness/judge_check", exist_ok=True)
with open(f"{R}/harness/judge_check/check_in.jsonl", "w") as f:
    for i, o in enumerate(out):
        o["check_id"] = i
        f.write(json.dumps(o) + "\n")
print(len(out), "items;", sum(o["phrase_refusal"] for o in out), "phrase refusals")
