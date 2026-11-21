#!/usr/bin/env bash
# Pod (15:45Z replan): subsets, finalists U=c1h_r1_svd1_s3_L4 and F=c1t013; c1h_svd1_s2 dropped.
# Stops workers (running bench children continue and are adopted), archives the old queue/claims, writes a new queue.
cd /srv/suite
for i in 0 1 2 3 4 5 6 7; do [ -f worker_$i.pid ] && kill $(cat worker_$i.pid) 2>/dev/null; done
sleep 2
mv queue.jsonl queue_v1.jsonl; mv claims.jsonl claims_v1.jsonl
python3 - <<'EOF'
import json
rows = []
def add(m, b, n, prio, **kw):
    for k in range(n):
        rows.append({"model": m, "bench": b, "shard": k, "nshards": n, "prio": prio, **kw})
# full-set runs already generating (orphans, adopted): keep their shard layout
for m, b in [("orig", "math500"), ("orig", "gsm8k"), ("ou", "gpqa_diamond"), ("ou", "math500")]:
    add(m, b, 2, 9)
U, F = "c1h_r1_svd1_s3_L4", "c1t013"
add(U, "mmlu_pro", 1, 10)
add("orig", "mmlu_pro", 1, 8); add("ou", "mmlu_pro", 1, 7); add(F, "mmlu_pro", 1, 6)
for m in ["orig", U, "ou", F]:
    if m != "orig": add(m, "math500", 2, 5)
    add(m, "gpqa_diamond", 2, 5) if m != "ou" else None
    if m != "orig": add(m, "gsm8k", 2, 4)
    add(m, "ifeval", 1, 3)
    add(m, "humaneval_plus", 1, 3)
open("queue.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "jobs")
EOF
touch claims.jsonl
bash start_workers.sh orig orig orig orig ou ou ou ou
