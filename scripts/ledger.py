#!/usr/bin/env python3
"""Budget ledger. Writes state/budget.json and state/pods.json from `lium ps`.

spent     = final spend of removed pods (state/pods_removed.jsonl) + spent_usd of live pods
committed = spent + live pods' price × hours left until their scheduled removal
Run once, or with --loop to refresh every 10 minutes.
"""
import json, os, subprocess, sys, time
from datetime import datetime, timezone

ROOT = os.path.expanduser("~/Code/inkling-abl")
CAP = 200.0
ENV = {**os.environ, "LIUM_NONINTERACTIVE": "1"}


def live_pods() -> list[dict]:
    out = subprocess.run(["lium", "ps", "--format", "json"], capture_output=True, text=True, env=ENV, timeout=120)
    try:
        # The account also runs nokeep production pods; only this project's pods count.
        return [p for p in json.loads(out.stdout or "[]") if str(p.get("name", "")).startswith("ink-")]
    except json.JSONDecodeError:
        return []


def removed_spend() -> float:
    path = f"{ROOT}/state/pods_removed.jsonl"
    if not os.path.exists(path):
        return 0.0
    return sum(json.loads(l).get("final_spent_usd", 0.0) for l in open(path) if l.strip())


def hours_left(pod: dict) -> float:
    ts = pod.get("removal_scheduled_at")
    if not ts:
        return 2.0  # brief: TTL is never > 2h; assume worst case if unknown
    end = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    return max(0.0, (end - datetime.now(timezone.utc)).total_seconds() / 3600)


def refresh() -> dict:
    pods = live_pods()
    live_spent = sum(p.get("spent_usd", 0.0) for p in pods)
    pending = sum(p.get("price_per_hour", 0.0) * hours_left(p) for p in pods)
    spent = removed_spend() + live_spent
    ledger = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cap_usd": CAP,
        "spent_usd": round(spent, 2),
        "committed_usd": round(spent + pending, 2),
        "burn_per_hour": round(sum(p.get("price_per_hour", 0.0) for p in pods), 2),
        "live_pods": len(pods),
        "thresholds": {"notice": 120, "no_new_pods": 170, "evacuate": 195},
    }
    slim = [{k: p.get(k) for k in ("name", "huid", "gpu_type", "price_per_hour", "spent_usd",
                                    "status", "removal_scheduled_at", "ip")} | {"ssh_port": (p.get("ports") or {}).get("22")}
            for p in pods]
    json.dump(ledger, open(f"{ROOT}/state/budget.json", "w"), indent=1)
    json.dump(slim, open(f"{ROOT}/state/pods.json", "w"), indent=1)
    return ledger


if __name__ == "__main__":
    if "--loop" in sys.argv:
        while True:
            print(json.dumps(refresh()), flush=True)
            time.sleep(600)
    print(json.dumps(refresh(), indent=1))
