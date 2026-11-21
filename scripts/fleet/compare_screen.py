"""Compare a fleet run against parity's run of the same candidate on the same prompts (phrase scorer, same answers?).
  python3 compare_screen.py <fleet_outputs_dir> <parity_eval_dir>
For each gens file in the fleet dir: refusal rate on the shared ids for both, refusal disagreements, identical answers.
Plus MMLU: accuracy both ways and per-item prediction agreement."""
import json
import os
import sys

f_dir, p_dir = sys.argv[1], sys.argv[2]


def read(p: str) -> dict:
    return {r["id"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


for name in ("harmful_off", "harmful_e09", "gsm8k"):
    a, b = read(f"{f_dir}/gens_{name}.jsonl"), read(f"{p_dir}/gens_{name}.jsonl")
    ids = sorted(set(a) & set(b))
    if not ids:
        continue
    key = "refusal" if name.startswith("harmful") else "correct"
    ra = sum(float(a[i][key]) for i in ids) / len(ids)
    rb = sum(float(b[i][key]) for i in ids) / len(ids)
    dis = sum(a[i][key] != b[i][key] for i in ids)
    same = sum(a[i]["answer"] == b[i]["answer"] for i in ids)
    print(f"{name}: n={len(ids)} fleet {key}={ra:.4f} parity {key}={rb:.4f} disagree={dis} identical_answers={same}")
a, b = read(f"{f_dir}/gens_mmlu.jsonl"), read(f"{p_dir}/gens_mmlu.jsonl")
ids = sorted(set(a) & set(b), key=str)
if ids:
    acc = lambda d: 100 * sum(d[i]["pred"] == d[i]["answer"] for i in ids) / len(ids)  # noqa: E731
    print(f"mmlu: n={len(ids)} fleet {acc(a):.2f} parity {acc(b):.2f} pred_agree={sum(a[i]['pred'] == b[i]['pred'] for i in ids)}")
