"""Pod: append core-suite jobs for the given models to /srv/suite/queue.jsonl.  python3 enqueue_core.py orig ou ..."""
import fcntl
import json
import sys

SHARDS = {"gpqa_diamond": (2, 5), "math500": (2, 4), "gsm8k": (2, 3), "ifeval": (1, 3), "mmlu_pro": (2, 2),
          "humaneval_plus": (1, 1)}
with open("/srv/suite/queue.jsonl", "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    for m in sys.argv[1:]:
        for b, (n, prio) in SHARDS.items():
            for k in range(n):
                f.write(json.dumps({"model": m, "bench": b, "shard": k, "nshards": n, "prio": prio}) + "\n")
print("enqueued", sys.argv[1:])
