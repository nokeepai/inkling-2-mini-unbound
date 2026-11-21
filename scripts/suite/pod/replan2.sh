#!/usr/bin/env bash
# Pod (replan 2): F = c1dt001 replaces c1t013. Stop workers (children continue, adopted), kill c1t013 jobs,
# rewrite the queue (claims kept, so finished jobs stay done), restart workers with true per-GPU model state.
cd /srv/suite
for i in 0 1 2 3 4 5 6 7; do [ -f worker_$i.pid ] && kill $(cat worker_$i.pid) 2>/dev/null; done
pkill -f "bench_core.py .* --model c1t013 " ; sleep 2
# wait for any in-flight container restart, then record what each container serves
while pgrep -f suite_serve.sh >/dev/null; do sleep 10; done
for i in 0 1 2 3 4 5 6 7; do
  d=$(docker inspect -f '{{ index .Config.Labels "sig" }}' s$i | cut -d'|' -f1); m=${d#/w/}; m=${m#merged_}
  echo -n $m > gpu$i.model; echo "g$i serves $m"
done
mv queue.jsonl queue_v2.jsonl
python3 - <<'EOF'
import json
rows = []
def add(m, b, n, prio):
    for k in range(n):
        rows.append({"model": m, "bench": b, "shard": k, "nshards": n, "prio": prio})
for m, b in [("orig", "math500"), ("orig", "gsm8k"), ("ou", "gpqa_diamond"), ("ou", "math500")]:
    add(m, b, 2, 9)
U, F = "c1h_r1_svd1_s3_L4", "c1dt001"
add(U, "mmlu_pro", 1, 10)
add("orig", "mmlu_pro", 1, 8); add("ou", "mmlu_pro", 1, 7); add(F, "mmlu_pro", 1, 6)
for m in ["orig", U, "ou", F]:
    if m != "orig": add(m, "math500", 2, 5)
    if m != "ou": add(m, "gpqa_diamond", 2, 5)
    if m != "orig": add(m, "gsm8k", 2, 4)
    add(m, "ifeval", 1, 3)
    add(m, "humaneval_plus", 1, 3)
done = {json.loads(l)["key"] for l in open("claims.jsonl") if '"done"' in l}
# errored/claimed-but-not-done keys from replan 1 must run again: give them try 1
errs = {json.loads(l)["key"] for l in open("claims.jsonl")} - done
for r in rows:
    k = f"{r['model']}|{r['bench']}|{r['shard']}/{r['nshards']}"
    if k in errs:
        r["try"] = 1
open("queue.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "jobs;", len(done), "already done")
EOF
bash start_workers.sh x x x x x x x x
