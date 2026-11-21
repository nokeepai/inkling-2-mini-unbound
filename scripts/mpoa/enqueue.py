"""Append queue rows for mpoa candidates.  enqueue.py <stage> <priority> <family> <cand_id>..."""
import fcntl
import json
import os
import sys
from datetime import datetime, timezone

R = os.path.expanduser("~/Code/inkling-abl")


def enqueue(cids: list[str], stage: str, priority: float, family: str, note: str = "") -> None:
    with open(f"{R}/state/queue.jsonl", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for cid in cids:
            assert os.path.exists(f"{R}/methods/mpoa/cands/{cid}/lowrank.safetensors"), cid
            row = {"cand_id": cid, "family": family, "path": f"methods/mpoa/cands/{cid}", "priority": priority,
                   "stage": stage, "node": "fleet", "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "by": "mpoa"}
            if note:
                row["note"] = note
            f.write(json.dumps(row) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    stage, prio, fam = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    enqueue(sys.argv[4:], stage, prio, fam)
    print("enqueued", len(sys.argv[4:]))
