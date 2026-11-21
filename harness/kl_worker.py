"""KL worker, control-machine loop (methods/SEARCH.md §Loop). Drives the long-lived pod server harness/kl_server.py
on ink-research-2 through lium exec / rsync. W4A16 only.

Every 15 s:
1. Read R/state/queue.jsonl ({"cand_id","family","path","priority","time"}). Pending = no `kl` / `kl_error` row from
   agent kl_worker for that cand_id in results/runs.jsonl, and not claimed by another KL worker.
2. Claim the highest-priority, oldest pending candidate: append {"claim","worker","kind":"kl","time"} to
   R/state/queue_claims.jsonl under flock (re-checking claims inside the lock).
3. rsync the folder to /srv/cands/<cand_id>/ on the pod, verify the payload sha256 on the pod, write READY with a nonce.
4. Wait for the server's row with that nonce. The server swaps the tensors in (bit-verified), scores against the cache
   with its exact batching, restores, and checks a probe forward reproduces the startup logits exactly.
5. Append rows `kl` and `top1` (agent "kl_worker", split "val", n, CI, engine "hf_w4a16", candidate = cand_id), or one
   `kl_error` row. For compact candidates without sha256.txt, the canonical sha from the pod build is written back.
Status: R/state/agents/kl_worker.txt every 10 min and on each result. Log: R/harness/fidelity/kl_worker.log.

  python3 R/harness/kl_worker.py           (loop)
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

R = os.path.expanduser("~/Code/inkling-abl")
POD = os.environ.get("KL_POD", "ink-research-2b")
WORKER = f"kl_worker@{POD}"
# Claims by earlier incarnations of this worker (pod replaced at 12:28Z) count as ours, so they are not skipped.
MINE = {WORKER, "kl_worker@ink-research-2"}
# The reference cache the server holds. Rows carry it; a candidate counts as scored only for the current cache.
OLD_CACHE = "894ba2e64a44985369adbe678a5ef914a8d3ee9fbcc2b7b0c0530fd03190684d"
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fidelity", "cache_sha.txt")
QUEUE = f"{R}/state/queue.jsonl"
CLAIMS = f"{R}/state/queue_claims.jsonl"
RUNS = f"{R}/results/runs.jsonl"
STATUS = f"{R}/state/agents/kl_worker.txt"
RESULTS = "/root/fid/kl_server_results.jsonl"
BUILD_LOCAL = f"{R}/harness/build_cand.py"
POLL, WAIT_MAX = 15, 900
ENV = {**os.environ, "LIUM_NONINTERACTIVE": "1"}
GUARD = f"{R}/scripts/ink_lium.sh"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"{now()} {msg}", flush=True)


def lium(*args: str, timeout: int = 900) -> str:
    # Owner rule: every pod-targeting lium call goes through the guarded wrapper (refuses non-ink-* targets).
    r = subprocess.run(["bash", GUARD, *args], capture_output=True, text=True, env=ENV, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ink_lium {' '.join(args)[:150]} failed: {(r.stderr or r.stdout)[-400:]}")
    return r.stdout


def remote(cmd: str) -> str:
    out = lium("exec", POD, cmd)
    return "\n".join(l for l in out.splitlines() if not l.startswith("Executing on"))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path):
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    return out


def cache_sha() -> str:
    return open(CACHE_FILE).read().split()[0]


def scored() -> set[str]:
    """Candidates with a `kl` row for the current cache, or a server-side `kl_error` row. Transport-error rows
    (written before 12:45Z when the pod expired; they say "after 3 tries") do not count."""
    cur = cache_sha()
    out = set()
    for r in read_jsonl(RUNS):
        if r.get("agent") != "kl_worker":
            continue
        if r.get("metric") == "kl" and r.get("cache", OLD_CACHE) == cur:
            out.add(r["candidate"])
        elif r.get("metric") == "kl_error" and "after 3 tries" not in str(r.get("error")) and r.get("cache") == cur:
            out.add(r["candidate"])
    return out


def kl_claimed_by_others(claims: list[dict]) -> set[str]:
    return {c["claim"] for c in claims if str(c.get("worker", "")).startswith("kl_worker") and c.get("worker") not in MINE}


def my_claims(claims: list[dict]) -> set[str]:
    return {c["claim"] for c in claims if c.get("worker") in MINE}


SKIPPED: set[str] = set()
SKIP_LOGGED: set[str] = set()


def skip_reason(e: dict) -> str | None:
    """Queue rows the KL worker never scores: eval-only rows, and folders with no tensors and no lowrank file
    (e.g. the fleet's `orig` baseline)."""
    if e.get("kind") == "eval_only" or e.get("cand_id") == "orig":  # fleet baselines; orig's old row has no kind
        return "kind eval_only"
    d = resolve(e)
    if not any(os.path.exists(os.path.join(d, f)) for f in ("tensors.safetensors", "lowrank.safetensors")):
        return f"no tensors.safetensors or lowrank.safetensors in {d}"
    return None


def pending() -> list[dict]:
    done = scored()
    claims = read_jsonl(CLAIMS)
    taken = kl_claimed_by_others(claims)
    q = {}
    for e in read_jsonl(QUEUE):
        if "cand_id" in e:
            q[e["cand_id"]] = e  # last entry wins
    todo = []
    for c, e in q.items():
        if c in done or c in taken or c in SKIPPED:
            continue
        why = skip_reason(e)
        if why:
            if why == "kind eval_only":
                SKIPPED.add(c)
            if c not in SKIP_LOGGED:  # log once; a folder without files is re-checked (files may still arrive)
                SKIP_LOGGED.add(c)
                log(f"skip {c}: {why} (no row)")
            continue
        todo.append(e)
    return sorted(todo, key=lambda e: (-float(e.get("priority", 0) or 0), str(e.get("time", ""))))


def claim(cid: str) -> bool:
    os.makedirs(os.path.dirname(CLAIMS), exist_ok=True)
    with open(CLAIMS, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        claims = [json.loads(l) for l in f if l.strip()]
        if cid in kl_claimed_by_others(claims):
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
        if cid not in my_claims(claims):
            f.write(json.dumps({"claim": cid, "worker": WORKER, "kind": "kl", "time": now()}) + "\n")
            f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return True


def resolve(e: dict) -> str:
    p = e.get("path") or f"methods/{e['family']}/cands/{e['cand_id']}"
    p = p if os.path.isabs(p) else os.path.join(R, p)
    return p.rstrip("/")


def payload(d: str) -> str:
    """Compact form wins: if lowrank.safetensors exists, only the small files are pushed and the pod builds."""
    for f in ("lowrank.safetensors", "tensors.safetensors"):
        if os.path.exists(os.path.join(d, f)):
            return f
    raise FileNotFoundError(f"{d}: no tensors.safetensors or lowrank.safetensors")


def sync_build_script(state: dict) -> None:
    if os.path.exists(BUILD_LOCAL):
        h = sha256(BUILD_LOCAL)
        if state.get("build_sha") != h:
            lium("scp", POD, BUILD_LOCAL, "/root/fid/build_cand.py")
            state["build_sha"] = h


def submit(cid: str, d: str) -> str:
    pf = payload(d)
    local_sha = sha256(os.path.join(d, pf))
    rd = f"/srv/cands/{cid}"
    remote(f"rm -rf {rd}; mkdir -p {rd}")
    if pf == "lowrank.safetensors":
        # small files only; the canonical sha256.txt is written by the research-2 build, never uploaded
        lium("rsync", POD, d + "/", rd + "/", "--exclude", "tensors.safetensors", "--exclude", "sha256.txt",
             "--exclude", "built_sha256.txt")
    else:
        lium("rsync", POD, d + "/", rd + "/")
    got = remote(f"sha256sum {rd}/{pf}")
    if local_sha not in got:
        # rsync layout fallback: some wrappers nest the folder; flatten and re-check
        remote(f"[ -d {rd}/{os.path.basename(d)} ] && cp -a {rd}/{os.path.basename(d)}/. {rd}/ || true")
        got = remote(f"sha256sum {rd}/{pf}")
        if local_sha not in got:
            raise RuntimeError(f"{cid}: payload sha mismatch on pod ({got.strip()[:80]})")
    nonce = uuid.uuid4().hex
    remote(f"echo {nonce} > {rd}/READY")
    return nonce


def wait_result(cid: str, nonce: str) -> dict:
    t0 = time.time()
    while time.time() - t0 < WAIT_MAX:
        out = remote(f"grep -F '{nonce}' {RESULTS} || true")
        for l in out.splitlines():
            if nonce in l:
                return json.loads(l)
        st = remote("cat /root/fid/kl_server.status 2>/dev/null || true")
        if "FATAL" in st:
            raise RuntimeError(f"kl_server fatal: {st.strip()}")
        time.sleep(5)
    raise TimeoutError(f"{cid}: no result within {WAIT_MAX}s")


def append_rows(cid: str, r: dict) -> None:
    base = {"agent": "kl_worker", "candidate": cid, "split": "val", "pod": POD, "n": r.get("n"),
            "engine": "hf_w4a16", "act_quant": False, "sha256": r.get("sha256"), "bs": r.get("bs"),
            "order": r.get("order"), "sec": r.get("total_sec"), "restore_probe_ok": r.get("restore_probe_ok"),
            "cache": cache_sha()}
    ts = datetime.now(timezone.utc).strftime("%H%M%S")
    if "kl" in r:
        rows = [{**base, "run_id": f"klw-{cid}-kl-{ts}", "metric": "kl", "value": r["kl"], "CI": r["kl_ci"],
                 "n_pos": r["n_pos"], "pos_stats": r["pos_stats"], "probe_changed": r.get("probe_changed")},
                {**base, "run_id": f"klw-{cid}-top1-{ts}", "metric": "top1", "value": r["top1"], "CI": r["top1_ci"],
                 "n_pos": r["n_pos"]}]
    else:
        rows = [{**base, "run_id": f"klw-{cid}-err-{ts}", "metric": "kl_error", "value": None,
                 "error": r.get("error") or r.get("restore_error")}]
    subprocess.run([sys.executable, f"{R}/scripts/append_run.py"], input="\n".join(json.dumps(x) for x in rows),
                   text=True, check=True)


def record_canonical(d: str, sha: str) -> None:
    """Compact candidates: the research-2 build's sha is canonical. A different local sha256.txt (e.g. from a local
    build) is kept as sha256.local.txt and logged."""
    p = os.path.join(d, "sha256.txt")
    if os.path.exists(p):
        old = open(p).read().split()[0]
        if old == sha:
            return
        os.replace(p, os.path.join(d, "sha256.local.txt"))
        log(f"{os.path.basename(d)}: local sha {old[:12]} != canonical research-2 sha {sha[:12]} (kept as sha256.local.txt)")
    open(p, "w").write(sha + "  tensors.safetensors (canonical, built on ink-research-2)\n")


def status(msg: str) -> None:
    open(STATUS, "w").write(f"{datetime.now(timezone.utc).strftime('%H:%MZ')} kl_worker ({POD}): {msg}\n")


def process(e: dict, state: dict) -> None:
    cid = e["cand_id"]
    if not claim(cid):
        return
    d = resolve(e)
    t0 = time.time()
    try:
        nonce = submit(cid, d)
        r = wait_result(cid, nonce)
    except Exception as ex:  # noqa: BLE001 - transport/pod errors: never a row; log, back off, retry later
        tries = state["tries"].get(cid, 0) + 1
        state["tries"][cid] = tries
        log(f"{cid} transport error (try {tries}): {type(ex).__name__}: {ex}")
        status(f"transport error on {cid} (try {tries}): {ex}"[:200])
        time.sleep(min(300, 15 * tries))
        return
    if r.get("sha256") and os.path.exists(os.path.join(d, "lowrank.safetensors")):
        record_canonical(d, r["sha256"])
    append_rows(cid, r)
    state["n"] += 1
    msg = (f"{cid} KL {r['kl']:.5f} top1 {r['top1']:.4f} restore_ok={r.get('restore_probe_ok')}"
           if "kl" in r else f"{cid} ERROR {r.get('error')}")
    log(f"{msg} ({time.time() - t0:.0f}s end to end)")
    status(f"last: {msg}; scored {state['n']} this session")
    if "kl" in r and not r.get("restore_probe_ok"):
        raise SystemExit("restore probe failed on the pod; stopping")


def main() -> None:
    state = {"n": 0, "build_sha": None, "last_status": 0.0, "tries": {}}
    log(f"kl_worker loop start ({WORKER})")
    while True:
        try:
            sync_build_script(state)
            todo = pending()
            if todo:
                process(todo[0], state)
                time.sleep(1)
                continue
            if time.time() - state["last_status"] > 600:
                st = remote("cat /root/fid/kl_server.status 2>/dev/null || true").strip()
                status(f"idle, queue empty; scored {state['n']} this session; server: {st[-120:]}")
                state["last_status"] = time.time()
        except SystemExit:
            raise
        except Exception as ex:  # noqa: BLE001 - keep looping, but say so
            log(f"loop error: {type(ex).__name__}: {ex}")
            status(f"loop error: {ex}"[:200])
        time.sleep(POLL)


if __name__ == "__main__":
    main()
