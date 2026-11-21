"""Promote out-of-study mpoa candidates (hand seeds c1h*, and old-study c1t/c2t orphans) from screen to full.
Runs until the stop time; one full enqueue per candidate (state/optuna/promoted.txt)."""
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
import enqueue as Q
import metrics as MET

R = os.path.expanduser("~/Code/inkling-abl")
DONE = f"{R}/state/optuna/promoted.txt"
PREFIXES = ["c1h", "c1t0", "c2t0"]
while datetime.now(timezone.utc).strftime("%H:%M") < "15:00":
    done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
    for cid, m in MET.collect(PREFIXES, exclude_pods=("ink-eval-2",)).items():
        if cid in done or "kl" not in m or "screen_refusal_off" not in m:
            continue
        if m["screen_refusal_off"] <= 0.05 and m["kl"] <= 0.06:
            Q.enqueue([cid], "full", 7.0, "c1" if cid.startswith("c1") else "c2", note="promoted")
            open(DONE, "a").write(cid + "\n")
            print(datetime.now(timezone.utc).strftime("%H:%M:%S"), "promoted", cid, m["screen_refusal_off"], m["kl"], flush=True)
    time.sleep(30)
