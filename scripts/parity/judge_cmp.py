"""Compare judge outputs: GPU worker (queue out file) vs CPU reference (judge_check/check_out.jsonl)."""
import json
import sys

cpu = {json.loads(l)["check_id"]: json.loads(l) for l in open(
    "/Users/nikshepsvn/Code/inkling-abl/harness/judge_check/check_out.jsonl")}
gpu = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
agree = sum(int(cpu[k]["refused"] == gpu[k]["refused"]) for k in gpu)
same_score = sum(int(cpu[k]["judge_score"] == gpu[k]["judge_score"]) for k in gpu)
dmax = max(abs(cpu[k]["score"] - gpu[k]["score"]) for k in gpu)
print(f"refused agree {agree}/{len(gpu)}; same 1-5 score {same_score}/{len(gpu)}; max |score diff| {dmax:.3f}; "
      f"refused gpu {sum(g['refused'] for g in gpu.values())} cpu {sum(cpu[k]['refused'] for k in gpu)}")
