"""Queue-driven SGLang eval worker, Mac side (agent `fleet`). One process per RTX 5090 pod.

  python3 ~/Code/inkling-abl/harness/eval_worker.py --pod ink-eval-2 [--once] [--port 3001]

Loop: read R/state/queue.jsonl -> claim one eval job under flock on R/state/queue_claims.jsonl -> push the candidate's
small files to /srv/cands/<cand_id>/ (lowrank.safetensors preferred; built on the pod by build_cand.py and checked
against sha256.txt) -> run harness/fleet_eval.py on the pod under nohup (swap via sgl_eval's verified swap, score,
restore + verify) -> download outputs to R/results/outputs/<cand_id>/<tag>/ -> append metric rows to runs.jsonl
(agent "eval_worker:<pod>", split "val") -> append a "done" row to the claims file.

Queue row fields used: cand_id, path (relative to R), priority (higher first, then time), stage ("screen"|"full",
default "full"), finalist (bool). Job key = "<cand_id>|<stage>[|finalist]", so a screened candidate can later be
enqueued again as full.
Claims file rows (shared with the KL worker, which uses kind "kl"): {"claim"|"release"|"done": cand_id, "kind": "eval",
"key", "worker", "time", ...}. A key is open when it has no "done" row and no unreleased claim; after 2 error
releases nobody claims it again.
On any error: restore orig on the pod, append an eval_error row, release the claim, continue.
Heartbeat: R/state/agents/fleet_<pod>.txt.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone

R = os.path.expanduser("~/Code/inkling-abl")
QUEUE = f"{R}/state/queue.jsonl"
CLAIMS = f"{R}/state/queue_claims.jsonl"
OUTPUTS = f"{R}/results/outputs"
ENV = {**os.environ, "LIUM_NONINTERACTIVE": "1"}
INK_LIUM = f"{R}/scripts/ink_lium.sh"
ENGINE = "sglang_5090_frozen"
MAX_ERRORS = 2
CANON_POD = "ink-research-2"  # its KL worker builds compact candidates first; that build's sha is canonical
STALE_SEC =2 * 3600  # a full + finalist eval takes ~25 min; 2 h means the claiming worker died
# Files the pod needs in /srv/harness (relative to R/harness), and val data in /srv/data/val.
HARNESS_FILES = ("sgl_eval.py", "sgl_merge.py", "fleet_eval.py", "build_cand.py", "screen_ids.json", "borderline200_ids.json",
                 "refusal_phrases.txt", "refusal_phrases_seed.txt", "cap/mmlu300.jsonl", "cap/gsm8k100.jsonl")
DATA_FILES = ("harmful", "borderline", "probes")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lium(*args: str, timeout: int = 900) -> str:
    # owner rule: every pod-targeting call goes through the guarded wrapper (ink-* names only, no all/-a/row numbers)
    r = subprocess.run(["bash", INK_LIUM, *args], capture_output=True, text=True, env=ENV, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"lium {' '.join(args)[:160]} failed: {(r.stderr or r.stdout)[-500:]}")
    return r.stdout


def pod_exec(pod: str, cmd: str, timeout: int = 900) -> str:
    out = lium("exec", pod, cmd, timeout=timeout)
    # lium prints an "Executing on <huid>" banner first
    return "\n".join(l for l in out.splitlines() if not l.startswith("Executing on "))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def ready(job: dict) -> bool:
    """A compact candidate waits until the KL worker has recorded research-2's canonical sha256.txt next to it."""
    if job["cand_id"] == "orig":
        return True
    src = os.path.join(R, job["path"])
    if os.path.exists(f"{src}/lowrank.safetensors"):
        return os.path.exists(f"{src}/sha256.txt")
    return os.path.exists(f"{src}/tensors.safetensors") and os.path.exists(f"{src}/sha256.txt")


class Worker:
    def __init__(self, pod: str, port: int, gpu: int | None) -> None:
        self.pod, self.gpu = pod, gpu
        # multi-GPU pod: container sgl<gpu> on port 3001+gpu, per-GPU cand/merged dirs (fleet_eval.py --gpu)
        self.target = f"--gpu {gpu}" + (f" --port {port}" if port else "") if gpu is not None else f"--port {port or 3001}"
        self.slot = f"__g{gpu}" if gpu is not None else ""
        where = pod + (f":g{gpu}" if gpu is not None else "")
        self.where = where
        # queue rows carry an optional "node" ("fleet" default, or e.g. "eval-2"); a worker claims only its class
        self.node = "fleet" if pod.startswith("ink-fleet") else pod.removeprefix("ink-")
        self.name = f"eval_worker:{where}"
        self.status_path = f"{R}/state/agents/fleet_{pod}{'_g%d' % gpu if gpu is not None else ''}.txt"
        self.skip: set[str] = set()  # keys this process failed on; other workers may still retry
        self.done_count = 0
        self.sec: list[float] = []

    # ---- status ----
    def beat(self, msg: str) -> None:
        avg = f"; avg {sum(self.sec) / len(self.sec):.0f}s/cand" if self.sec else ""
        line = f"{datetime.now(timezone.utc).strftime('%H:%MZ')} fleet {self.where}: {msg} (done {self.done_count}{avg})"
        open(self.status_path, "w").write(line[:300] + "\n")
        print(line, flush=True)

    # ---- queue ----
    @staticmethod
    def job_key(row: dict) -> str:
        k = f"{row['cand_id']}|{row.get('stage', 'full')}"
        if row.get("finalist"):
            k += "|finalist"
        if int(row.get("think_cap", 1536)) != 1536:  # hygiene reruns at another cap are separate jobs
            k += f"|cap{int(row['think_cap'])}"
        return k

    def claim_next(self) -> dict | None:
        rows = [json.loads(l) for l in open(QUEUE) if l.strip()] if os.path.exists(QUEUE) else []
        jobs = [r for r in rows if "cand_id" in r and ("path" in r or r["cand_id"] == "orig")]
        jobs.sort(key=lambda r: (-float(r.get("priority", 0) or 0), r.get("time", "")))
        with open(CLAIMS, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                state: dict[str, dict] = {}
                for l in f:
                    try:
                        c = json.loads(l)
                    except json.JSONDecodeError:
                        continue
                    if c.get("kind") != "eval" or "key" not in c:
                        continue
                    s = state.setdefault(c["key"], {"claimed": False, "done": False, "errors": 0})
                    if "claim" in c:
                        # a claim older than STALE_SEC without done/release is a dead worker's; reopen it
                        age = time.time() - datetime.fromisoformat(c["time"]).timestamp()
                        s["claimed"] = age < STALE_SEC
                    elif "release" in c:
                        s["claimed"] = False
                        s["errors"] += int(bool(c.get("error")))
                    elif "done" in c:
                        s["done"] = True
                for j in jobs:
                    k = self.job_key(j)
                    s = state.get(k, {"claimed": False, "done": False, "errors": 0})
                    if j.get("node", "fleet") != self.node:
                        continue
                    if s["done"] or s["claimed"] or s["errors"] >= MAX_ERRORS or k in self.skip or not ready(j):
                        continue
                    f.write(json.dumps({"claim": j["cand_id"], "kind": "eval", "key": k, "worker": self.name,
                                        "time": now()}) + "\n")
                    f.flush()
                    return j
                return None
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def mark(self, verb: str, job: dict, **extra: object) -> None:
        with open(CLAIMS, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps({verb: job["cand_id"], "kind": "eval", "key": self.job_key(job), "worker": self.name,
                                "time": now(), **extra}) + "\n")
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

    # ---- pod I/O ----
    def ensure_harness(self) -> None:
        """Push harness code + val data when the pod copy differs (one sha256sum call, then only changed files).
        On a multi-GPU pod only the gpu-0 worker syncs, so 8 workers never overwrite files another one is reading."""
        if self.gpu not in (None, 0):
            return
        local = {f"/srv/harness/{f}": f"{R}/harness/{f}" for f in HARNESS_FILES if os.path.exists(f"{R}/harness/{f}")}
        local.update({f"/srv/data/val/{d}.jsonl": f"{R}/data/val/{d}.jsonl" for d in DATA_FILES})
        pod_exec(self.pod, "mkdir -p /srv/harness/cap /srv/data/val /srv/evals /srv/cands /srv/fleet /root/logs")
        remote = pod_exec(self.pod, "sha256sum " + " ".join(local) + " 2>/dev/null || true")
        have = {p: h for h, p in (l.split() for l in remote.splitlines() if len(l.split()) == 2)}
        for rp, lp in local.items():
            if have.get(rp) != sha256(lp):
                lium("scp", self.pod, lp, rp)

    def push_cand(self, job: dict) -> str:
        """Upload the candidate's small files; returns the sha the pod must reproduce (or '' if none shipped)."""
        src = os.path.join(R, job["path"])
        cid = job["cand_id"]
        rd = f"/srv/cands/{cid}{self.slot}"
        lowrank = os.path.exists(f"{src}/lowrank.safetensors")
        files = ["spec.json", "sha256.txt", "lowrank.safetensors" if lowrank else "tensors.safetensors"]
        files = [f for f in files if os.path.exists(f"{src}/{f}")]
        if not lowrank and "tensors.safetensors" not in files:
            raise RuntimeError(f"candidate {cid} has neither lowrank.safetensors nor tensors.safetensors")
        canon = open(f"{src}/sha256.txt").read().split()[0] if os.path.exists(f"{src}/sha256.txt") else ""
        if not lowrank and sha256(f"{src}/tensors.safetensors") != canon:
            raise RuntimeError("local tensors.safetensors sha256 != sha256.txt (still being written?)")
        big = files[-1]
        local_big = sha256(f"{src}/{big}")
        remote = pod_exec(self.pod, f"mkdir -p {rd}; sha256sum {rd}/{big} 2>/dev/null || true")
        if local_big not in remote:
            pod_exec(self.pod, f"rm -rf {rd}; mkdir -p {rd}")
            for f in files:
                lium("scp", self.pod, f"{src}/{f}", f"{rd}/{f}")
            if local_big not in pod_exec(self.pod, f"sha256sum {rd}/{big}"):
                raise RuntimeError(f"upload sha mismatch for {cid}/{big}")
        else:
            for f in files[:-1]:
                lium("scp", self.pod, f"{src}/{f}", f"{rd}/{f}")
        if lowrank:
            self.materialise(cid, rd, canon)
        return canon

    def materialise(self, cid: str, rd: str, canon: str) -> None:
        """Build tensors on the pod; on a sha mismatch, log it and copy research-2's canonical tensors pod to pod."""
        out = pod_exec(self.pod, f"cd /srv/harness && python3 fleet_eval.py --build-only --cand {cid} {self.target}; "
                                 f"echo BUILD_EXIT=$?", timeout=1200)
        if "BUILD_EXIT=0" in out:
            return
        if "BUILD_EXIT=3" not in out:
            raise RuntimeError(f"build failed: {out[-600:]}")
        self.beat(f"{cid}: {out.strip().splitlines()[0][:160]}; pulling canonical tensors from {CANON_POD}")
        self.log_mismatch(cid, out.strip().splitlines()[0])
        lium("cp", f"{CANON_POD}:/srv/cands/{cid}/tensors.safetensors", f"{self.pod}:{rd}/tensors.safetensors",
             timeout=1800)
        if canon not in pod_exec(self.pod, f"sha256sum {rd}/tensors.safetensors"):
            raise RuntimeError(f"canonical tensors from {CANON_POD} do not match sha256.txt {canon[:12]}")

    def log_mismatch(self, cid: str, msg: str) -> None:
        with open(f"{R}/state/agents/fleet_sha_mismatch.jsonl", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps({"time": now(), "cand_id": cid, "worker": self.name, "msg": msg}) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    def run_remote(self, job: dict, tag: str) -> dict:
        cid, stage = job["cand_id"], job.get("stage", "full")
        fin = " --finalist" if job.get("finalist") else ""
        fin += f" --think-cap {int(job.get('think_cap', 1536))}"
        ed = f"/srv/evals/{cid}_{tag}"
        log = f"/root/logs/fleet_{cid}_{tag}.log"
        pid = f"/srv/fleet/{cid}_{tag}.pid"
        # `cd x; nohup ... &` (not `cd x && nohup ... &`, which backgrounds a subshell holding the ssh stdout open)
        pod_exec(self.pod, f"mkdir -p /srv/fleet; cd /srv/harness; nohup python3 fleet_eval.py --cand {cid} "
                           f"--stage {stage}{fin} --tag {tag} {self.target} > {log} 2>&1 < /dev/null & echo $! > {pid}")  # detach stdin, or lium exec waits
        t0 = time.time()
        while True:
            time.sleep(20)
            st = pod_exec(self.pod, f"cat {ed}/fleet_done.json 2>/dev/null || cat {ed}/fleet_error.json 2>/dev/null "
                                    f"|| (kill -0 $(cat {pid}) 2>/dev/null && echo RUNNING) "
                                    f"|| (echo DEAD; tail -n 20 {log})").strip()
            if st.startswith("{"):
                return json.loads(st)
            if st.startswith("DEAD"):
                return {"error": "fleet_eval died without a result: " + st[-800:]}
            self.beat(f"running {cid} ({stage}) {time.time() - t0:.0f}s")

    def fetch_outputs(self, cid: str, tag: str) -> str:
        dest = f"{OUTPUTS}/{cid}/{tag}"
        os.makedirs(dest, exist_ok=True)
        tgz = f"/srv/evals/{cid}_{tag}.tgz"
        pod_exec(self.pod, f"tar czf {tgz} -C /srv/evals {cid}_{tag}")
        lium("scp", self.pod, tgz, dest + "/", "-d")
        local = f"{dest}/{os.path.basename(tgz)}"
        # fleet_done.json is the judge relay's completion marker: extract it LAST, after every gens file is complete
        with tarfile.open(local) as t:
            members = [m for m in t.getmembers() if m.isfile() and "/" in m.name and ".." not in m.name]
            members.sort(key=lambda m: os.path.basename(m.name) == "fleet_done.json")
            for m in members:
                m.name = os.path.basename(m.name)
                t.extract(m, dest)
        os.remove(local)
        pod_exec(self.pod, f"rm -f {tgz}")
        return dest

    # ---- rows ----
    def append_rows(self, job: dict, tag: str, summary: dict, sha: str) -> int:
        base = {"run_id": f"fleet-{job['cand_id']}-{tag}", "agent": self.name, "candidate": job["cand_id"],
                "split": "val", "pod": self.pod, "gpu": self.gpu, "engine": ENGINE, "stage": job.get("stage", "full"),
                "finalist": bool(job.get("finalist")), "family": job.get("family"), "sha256": sha or None,
                "think_cap": int(job.get("think_cap", 1536)),
                "tag": tag, "sec": summary.get("total_sec"), "scorer": "phrase"}
        rows = [{**base, "metric": k, "value": v["value"], "CI": list(v["CI"]), "n": v["n"]}
                for k, v in summary["metrics"].items()]
        rows.append({**base, "metric": "eval_sec", "value": summary.get("total_sec"), "timing": summary.get("timing")})
        subprocess.run([sys.executable, f"{R}/scripts/append_run.py"], input="\n".join(json.dumps(r) for r in rows),
                       text=True, check=True, capture_output=True)
        return len(rows)

    def append_error(self, job: dict, tag: str, err: str) -> None:
        row = {"run_id": f"fleet-{job['cand_id']}-{tag}", "agent": self.name, "candidate": job["cand_id"],
               "metric": "eval_error", "value": None, "split": "val", "pod": self.pod, "gpu": self.gpu, "engine": ENGINE,
               "stage": job.get("stage", "full"), "error": err[:1500]}
        subprocess.run([sys.executable, f"{R}/scripts/append_run.py", json.dumps(row)], check=True, capture_output=True)

    # ---- one job ----
    def process(self, job: dict) -> None:
        cid, stage = job["cand_id"], job.get("stage", "full")
        tag = f"{stage}{'F' if job.get('finalist') else ''}{'C%d' % int(job['think_cap']) if int(job.get('think_cap', 1536)) != 1536 else ''}-{self.where.replace(':', '-')}-{datetime.now(timezone.utc).strftime('%H%M%S')}"
        t0 = time.time()
        self.beat(f"claimed {cid} ({stage})")
        try:
            sha = self.push_cand(job) if cid != "orig" else ""
            res = self.run_remote(job, tag)
            if "error" in res:
                raise RuntimeError(f"pod: {res['error']} (restored={res.get('restored')})"
                                   + (f" restore_error={res.get('restore_error')}" if res.get("restore_error") else ""))
            dest = self.fetch_outputs(cid, tag)
            summary = json.load(open(f"{dest}/metrics.json"))
            prep = summary.get("prep") or {}
            sha = prep.get("sha256") or sha
            n = self.append_rows(job, tag, summary, sha)
            self.mark("done", job, tag=tag, sec=round(time.time() - t0, 1), rows=n)
            self.done_count += 1
            self.sec.append(time.time() - t0)
            self.beat(f"done {cid} ({stage}) in {time.time() - t0:.0f}s, {n} rows")
        except Exception as e:  # noqa: BLE001 - restore, record, release, continue
            err = f"{type(e).__name__}: {e}"
            restored = "unknown"
            try:
                out = pod_exec(self.pod, f"cd /srv/harness && python3 fleet_eval.py --restore-only {self.target}",
                               timeout=600)
                restored = "yes" if "RESTORED_OK" in out else "no"
            except Exception as e2:  # noqa: BLE001
                restored = f"no ({e2})"[:300]
            pod_exec(self.pod, f"rm -rf /srv/w/merged_{cid}{self.slot}")
            self.append_error(job, tag, f"{err} | restore: {restored}")
            self.mark("release", job, error=err[:500], tag=tag)
            self.skip.add(self.job_key(job))
            self.beat(f"ERROR on {cid}: {err[:120]} | restore {restored}")
            if not restored.startswith("yes"):
                raise SystemExit(f"pod {self.pod} could not be restored to orig; worker stopping") from e

    def loop(self, once: bool) -> None:
        self.ensure_harness()
        probe = pod_exec(self.pod, f"cd /srv/harness && python3 fleet_eval.py --probe-only {self.target}")
        self.beat(f"started; probe {probe.strip()[-160:]}")
        last_sync = time.time()
        while True:
            job = self.claim_next()
            if job is None:
                if once:
                    return
                self.beat("idle, queue empty")
                time.sleep(30)
                if time.time() - last_sync > 300:
                    self.ensure_harness()
                    last_sync = time.time()
                continue
            self.process(job)
            if once:
                return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", required=True)
    ap.add_argument("--port", type=int, default=0, help="host port on the pod (default 3001, or 3001+gpu)")
    ap.add_argument("--gpu", type=int, default=None, help="GPU index on a multi-GPU pod (container sgl<gpu>)")
    ap.add_argument("--once", action="store_true", help="process at most one job, then exit")
    a = ap.parse_args()
    if not a.pod.startswith("ink-"):
        raise SystemExit("eval workers run only on ink-* pods")
    Worker(a.pod, a.port, a.gpu).loop(a.once)


if __name__ == "__main__":
    main()
