"""Pod-side queue worker, one per GPU (agent `suite`).

  nohup python3 suite_worker.py --gpu i > /root/logs/worker_i.log 2>&1 &

Queue /srv/suite/queue.jsonl rows: {"model", "bench", "shard", "nshards", "prio"} (job key model|bench|shard/nshards).
Claims/done in /srv/suite/claims.jsonl (flock). Picks the highest prio; among equal prio prefers the model its server
already serves. Model switch: POST /update_weights_from_disk to /w/orig, /w/merged_ou or /w/merged_<model>, then checks
the probe's top-20 logprobs: equal to that model's recorded reference (first load records it), different from orig's.
When the last shard of a (model, bench) finishes, the worker runs `score`.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suite_common as S  # noqa: E402

Q, C = "/srv/suite/queue.jsonl", "/srv/suite/claims.jsonl"
PROBE = "Explain how to pick a basic pin tumbler lock."
HERE = os.path.dirname(os.path.abspath(__file__))


def base(r: dict) -> str:
    return f"{r['model']}|{r['bench']}|{r['shard']}/{r['nshards']}"


def key(r: dict) -> str:
    """A retry is a new queue row with "try": n; the last row per base key wins."""
    return base(r) + (f"#{r['try']}" if r.get("try") else "")


def queue() -> list[dict]:
    last: dict = {}
    for r in rows(Q):
        last[base(r)] = r
    return list(last.values())


def model_dir(m: str) -> str:
    return "/w/orig" if m == "orig" else ("/w/merged_ou" if m == "ou" else f"/w/merged_{m}")


def rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def claim(worker: str, current: str | None) -> dict | None:
    with open(C, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        taken = {json.loads(l)["key"] for l in f if l.strip()}  # claimed or done: never re-run automatically
        gpu = int(worker[1:])
        todo = [r for r in queue() if key(r) not in taken and r.get("gpu", gpu) == gpu]  # "gpu": pinned job
        if not todo:
            return None
        todo.sort(key=lambda r: (-r.get("prio", 0), r["model"] != current))
        job = todo[0]
        f.write(json.dumps({"key": key(job), "state": "claimed", "worker": worker, "time": time.time()}) + "\n")
        f.flush()
        return job


def mark(job: dict, state: str, worker: str, **extra) -> None:
    with open(C, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({"key": key(job), "state": state, "worker": worker,
                            "time": time.time(), **extra}) + "\n")


def probe(port: int) -> list:
    ids = S.render(PROBE, 0) + [S.MSG_MODEL, S.TEXT]
    o = S.post(port, "/generate", {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                                   "return_logprob": True, "top_logprobs_num": 20, "logprob_start_len": len(ids) - 1})
    return [[round(x[0], 6), x[1]] for x in o["meta_info"]["output_top_logprobs"][0]]


def container_key(gpu: int) -> str:
    env = json.loads(subprocess.run(["docker", "inspect", "-f", "{{json .Config.Env}}", f"s{gpu}"],
                                    capture_output=True, text=True).stdout or "[]")
    return next((e.split("=", 1)[1] for e in env if e.startswith("SGLANG_ADMIN_API_KEY=")), "")


def load_model(port: int, gpu: int, m: str, log) -> None:
    """Swap via the API when the container's admin key is usable; otherwise (the first launch raced on
    /srv/admin_key and baked a key with a newline into the env) restart the container on the new model."""
    if container_key(gpu) == open("/srv/admin_key").read().strip():
        r = S.post(port, "/update_weights_from_disk", {"model_path": model_dir(m)})
        assert r.get("success", True) is not False, r
    else:
        log(f"restart s{gpu} on {model_dir(m)} (unusable admin key)")
        q = subprocess.run(["bash", "/srv/suite/suite_serve.sh", str(gpu), str(port), model_dir(m)],
                           capture_output=True, text=True)
        assert q.returncode == 0, q.stdout[-500:] + q.stderr[-500:]
    for _ in range(120):  # server must answer before we probe (restart or reload in progress)
        try:
            probe(port)
            break
        except Exception:  # noqa: BLE001
            time.sleep(5)
    p = probe(port)
    ref = f"/srv/suite/probe_{m}.json"
    if os.path.exists(ref):
        if json.load(open(ref)) != p:
            log(f"WARNING probe for {m} on gpu{gpu} differs from reference")
    else:
        json.dump(p, open(ref, "w"))
    if m != "orig" and os.path.exists("/srv/suite/probe_orig.json"):
        assert json.load(open("/srv/suite/probe_orig.json")) != p, f"{m}: probe equals orig (swap had no effect)"
    open(f"/srv/suite/gpu{gpu}.model", "w").write(m)


def shards_done(model: str, bench: str) -> bool:
    done = {json.loads(l)["key"] for l in open(C) if '"done"' in l}
    want = [key(r) for r in queue() if r["model"] == model and r["bench"] == bench]
    return bool(want) and all(k in done for k in want)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    port, worker = 4000 + a.gpu, f"g{a.gpu}"
    logf = open(f"/root/logs/worker_{a.gpu}.events", "a")

    def log(s: str) -> None:
        logf.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {s}\n")
        logf.flush()
    cur_path = f"/srv/suite/gpu{a.gpu}.model"
    if not os.path.exists("/srv/suite/probe_orig.json") and open(cur_path).read().strip() == "orig":
        json.dump(probe(port), open("/srv/suite/probe_orig.json", "w"))
    idle = 0
    while True:
        cur = open(cur_path).read().strip() if os.path.exists(cur_path) else None
        job = claim(worker, cur)
        if job is None:
            idle += 1
            if idle == 1:
                log("idle")
            time.sleep(20)
            continue
        idle = 0
        t = time.time()
        try:
            script = job.get("script", "bench_core.py")
            tail = ["--model", job["model"], "--shard", str(job["shard"]), "--nshards", str(job["nshards"])]
            pat = f"{script} {job['bench']} gen --port [0-9]+ " + " ".join(tail) + "$"
            gp = os.path.join(S.OUT, job["model"], job["bench"], f"gens_{job['shard']}of{job['nshards']}.jsonl")
            if os.path.exists(gp):  # written atomically at the end of a run (an orphaned child finished it)
                log(f"already generated {key(job)}")
            elif subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
                # an earlier worker's child is still generating this exact job: wait for it instead of rerunning
                log(f"adopt running {key(job)}")
                while subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
                    time.sleep(15)
                gp = os.path.join(S.OUT, job["model"], job["bench"], f"gens_{job['shard']}of{job['nshards']}.jsonl")
                assert os.path.exists(gp), f"adopted job ended without {gp}"
            else:
                busy = f"gen --port {port} "
                while subprocess.run(["pgrep", "-f", busy], capture_output=True).returncode == 0:
                    time.sleep(15)  # an orphaned job still uses this server: never swap/run under it
                if job["model"] != cur:
                    log(f"load {job['model']} (was {cur})")
                    load_model(port, a.gpu, job["model"], log)
                cmd = [sys.executable, os.path.join(HERE, script), job["bench"], "gen", "--port", str(port)] + tail
                log(f"start {key(job)}")
                p = subprocess.run(cmd, capture_output=True, text=True)
                open(f"/root/logs/job_{key(job).replace('|', '_').replace('/', 'of')}.log", "w").write(
                    p.stdout + p.stderr)
                if p.returncode != 0:
                    raise RuntimeError(p.stderr[-800:])
            mark(job, "done", worker, sec=round(time.time() - t))
            log(f"done {key(job)} {time.time() - t:.0f}s")
            if shards_done(job["model"], job["bench"]):
                q = subprocess.run([sys.executable, os.path.join(HERE, script), job["bench"], "score",
                                    "--model", job["model"]], capture_output=True, text=True)
                open(f"/root/logs/score_{job['model']}_{job['bench']}.log", "w").write(q.stdout + q.stderr)
                log(f"score {job['model']}|{job['bench']} rc={q.returncode} {q.stdout.strip()[-300:]}")
        except Exception as e:  # noqa: BLE001  loud: error row + log; retried once (try+1), then left for a human
            mark(job, "error", worker, err=str(e)[-800:])
            log(f"ERROR {key(job)}: {str(e)[-300:]}")
            if job.get("try", 0) < 2:
                with open(Q, "a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write(json.dumps(dict(job, **{"try": job.get("try", 0) + 1})) + "\n")
            time.sleep(10)


if __name__ == "__main__":
    main()
