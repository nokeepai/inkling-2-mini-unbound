"""Agreement + Cohen's kappa of 'phrase OR HarmBench-no' vs the orchestrator's refused labels (no tuning)."""
import json
import os

R = os.path.expanduser("~/Code/inkling-abl/harness/judge_check")
lab = {int(k): v for k, v in json.load(open(f"{R}/labels_orchestrator.json"))["labels"].items()}
gem = {r["check_id"]: r for r in map(json.loads, open(f"{R}/check_out.jsonl"))}
hb = {r["id"]: r for r in map(json.loads, open("/private/tmp/hb_out.jsonl"))}
ks = sorted(hb)
pred = [int(gem[k]["phrase_refusal"] or not hb[k]["hb_yes"]) for k in ks]
y = [lab[k]["refused"] for k in ks]
n = len(ks)
po = sum(int(a == b) for a, b in zip(pred, y)) / n
pe = (sum(pred) / n) * (sum(y) / n) + (1 - sum(pred) / n) * (1 - sum(y) / n)
print(f"phrase OR HB-no vs refused: {round(po * n)}/{n} ({100 * po:.0f}%) kappa {(po - pe) / (1 - pe):.2f}; "
      f"wrong={[k for k, a, b in zip(ks, pred, y) if a != b]}")
