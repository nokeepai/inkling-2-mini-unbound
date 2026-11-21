#!/usr/bin/env python3
"""TTL guard: every 5 min, for each live ink-* pod whose scheduled removal is < 25 min away
and which is still in use (listed in state/pods_keep.txt), reschedule removal to 2h from now
(the brief's max TTL). Pods not listed in pods_keep.txt are left to expire.
Logs to state/ttl_guard.log. Runs only while this session's Mac is up, so the TTL still bounds
spend if the orchestrator dies."""
import json, os, subprocess, time
from datetime import datetime, timezone

ROOT = os.path.expanduser("~/Code/inkling-abl")
ENV = {**os.environ, "LIUM_NONINTERACTIVE": "1", "LIUM_NO_POD_INDEX": "1"}


def log(msg: str) -> None:
    with open(f"{ROOT}/state/ttl_guard.log", "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}\n")


def keep_list() -> set[str]:
    path = f"{ROOT}/state/pods_keep.txt"
    return {l.strip() for l in open(path) if l.strip().startswith("ink-")} if os.path.exists(path) else set()


def tick() -> None:
    out = subprocess.run(["lium", "ps", "--format", "json"], capture_output=True, text=True, env=ENV, timeout=120)
    pods = [p for p in json.loads(out.stdout or "[]") if str(p.get("name", "")).startswith("ink-")]
    keep = keep_list()
    for p in pods:
        ts = p.get("removal_scheduled_at")
        if not ts:
            continue
        left = (datetime.fromisoformat(ts).replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).total_seconds() / 60
        if left < 25 and p["name"] in keep:
            r = subprocess.run(["bash", f"{ROOT}/scripts/pod_extend.sh", p["name"], "2h"], capture_output=True, text=True)
            log(f"extended {p['name']} ({left:.0f} min left): {r.stdout.strip()} {r.stderr.strip()}")
        elif left < 25:
            log(f"NOT extending {p['name']} ({left:.0f} min left; not in pods_keep.txt)")


if __name__ == "__main__":
    while True:
        try:
            tick()
        except Exception as e:  # keep guarding; log loudly
            log(f"ERROR {e!r}")
        time.sleep(300)
