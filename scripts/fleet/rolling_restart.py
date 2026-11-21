"""Restart the Mac-side eval workers of a pod onto new code without losing work. Per GPU, in parallel:
kill the worker (only the PID in scripts/fleet/pids/), let its pod job finish, collect it (rows + done), or release
its claim if no pod job was running, then start a fresh worker for that GPU.
  python3 rolling_restart.py ink-fleet-2 8"""
import json
import os
import subprocess
import sys
import threading
import time

R = os.path.expanduser("~/Code/inkling-abl")
sys.path.insert(0, f"{R}/harness")
import eval_worker as W  # noqa: E402

POD, N = sys.argv[1], int(sys.argv[2])
ONLY = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else list(range(N))


def last_open_claim(name: str) -> dict | None:
    """The worker's newest claim row with no later done/release for that key."""
    rows = [json.loads(l) for l in open(W.CLAIMS) if l.strip()]
    last = None
    for c in rows:
        if c.get("kind") != "eval" or c.get("worker") != name:
            continue
        if "claim" in c:
            last = c
        elif last and c.get("key") == last.get("key") and ("done" in c or "release" in c):
            last = None
    return last


def one(g: int) -> None:
    name = f"eval_worker:{POD}:g{g}"
    pidf = f"{R}/scripts/fleet/pids/{POD}_g{g}.pid"
    pid = int(open(pidf).read())
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        pass
    time.sleep(3)
    claim = last_open_claim(name)
    if claim:
        cid, key = claim["claim"], claim["key"]
        parts = key.split("|")
        stage, fin = parts[1], "finalist" in parts
        cap = next((int(p[3:]) for p in parts if p.startswith("cap")), 1536)
        d = W.pod_exec(POD, f"ls -dt /srv/evals/{cid}_*-g{g}-* 2>/dev/null | head -n 1").strip()
        claim_t = claim["time"]
        if d and "-" in d:
            tag = d.split(f"/srv/evals/{cid}_", 1)[1]
            while True:
                st = W.pod_exec(POD, f"ls {d}/fleet_done.json {d}/fleet_error.json 2>/dev/null; true").strip()
                if st:
                    break
                time.sleep(20)
            if "fleet_done" in st:
                args = ["python3", f"{R}/scripts/fleet/collect.py", POD, cid, tag, stage, "--gpu", str(g), "--think-cap", str(cap)]
                if fin:
                    args.append("--finalist")
                out = subprocess.run(args, capture_output=True, text=True)
                print(f"g{g}: collected {cid} {tag}: {out.stdout.strip()[-200:]} {out.stderr.strip()[-300:]}", flush=True)
            else:
                job = {"cand_id": cid, "stage": stage, "finalist": fin, "think_cap": cap}
                W.Worker(POD, 0, g).mark("release", job, error="pod error during rolling restart", tag=tag)
                print(f"g{g}: {cid} errored on pod; released", flush=True)
        else:
            job = {"cand_id": cid, "stage": stage, "finalist": fin, "think_cap": cap}
            W.Worker(POD, 0, g).mark("release", job, note=f"rolling restart before launch (claimed {claim_t})")
            print(f"g{g}: released unlaunched claim {key}", flush=True)
    if "--no-restart" in sys.argv:
        os.remove(pidf)
        print(f"g{g}: stopped (no restart); server sgl{g} serving orig", flush=True)
        return
    log = open(f"{R}/state/agents/fleet_{POD}_g{g}.log", "a")
    p = subprocess.Popen(["python3", f"{R}/harness/eval_worker.py", "--pod", POD, "--gpu", str(g)],
                         stdout=log, stderr=log, start_new_session=True)
    open(pidf, "w").write(str(p.pid))
    print(f"g{g}: restarted pid {p.pid}", flush=True)


threads = [threading.Thread(target=one, args=(g,)) for g in ONLY]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("ROLLING_RESTART_DONE")
