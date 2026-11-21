"""Mac side of the KL worker: push new candidates to ink-research-2, pull scores into runs.jsonl. Loops forever.

- Candidates: R/methods/*/cands/*/ with spec.json, tensors.safetensors and sha256.txt (sha must match the file).
  Skipped if runs.jsonl already has a fidelity kl_256 row with engine hf_w4a16 for that candidate id and sha
  (rows without a sha, e.g. the ou/c0 baselines, count for their candidate id).
- Upload to /srv/cands/<family>__<dir>/, verify sha256 on the pod, then write READY.
- Pull /root/fid/kl_worker_results.jsonl; append each new row (kl_256 + top1_256) via append_run.py.
State: R/harness/fidelity/kl_sync_state.json. Status: R/state/agents/fidelity.txt.

  python3 kl_sync.py            (loop)      python3 kl_sync.py --once
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

R = os.path.expanduser("~/Code/inkling-abl")
POD = "ink-research-2"
STATE = f"{R}/harness/fidelity/kl_sync_state.json"
RUNS = f"{R}/results/runs.jsonl"
SCRATCH = f"{R}/harness/fidelity/.sync"
ENV = {**os.environ, "LIUM_NONINTERACTIVE": "1"}


def lium(*args: str) -> str:
    r = subprocess.run(["lium", *args], capture_output=True, text=True, env=ENV, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"lium {' '.join(args)[:120]} failed: {r.stderr[-400:]}")
    return r.stdout


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def load_state() -> dict:
    return json.load(open(STATE)) if os.path.exists(STATE) else {"uploaded": {}, "appended": []}


def scored_in_runs() -> set[tuple[str, str | None]]:
    out = set()
    if os.path.exists(RUNS):
        for l in open(RUNS):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("agent") == "fidelity" and r.get("metric") == "kl_256" and r.get("engine") == "hf_w4a16":
                out.add((r["candidate"], r.get("sha256")))
    return out


def local_cands() -> list[dict]:
    out = []
    for d in sorted(glob.glob(f"{R}/methods/*/cands/*/")):
        files = [os.path.join(d, f) for f in ("spec.json", "tensors.safetensors", "sha256.txt")]
        if not all(os.path.exists(f) for f in files):
            continue
        want = open(files[2]).read().split()[0]
        spec = json.load(open(files[0]))
        family = d.rstrip("/").split("/")[-3]
        out.append({"dir": d, "key": f"{family}__{os.path.basename(d.rstrip('/'))}", "sha": want,
                    "cand_id": spec.get("cand_id"), "files": files})
    return out


def push(c: dict) -> None:
    if sha256(c["files"][1]) != c["sha"]:
        return  # still being written; next round
    rd = f"/srv/cands/{c['key']}"
    lium("exec", POD, f"rm -rf {rd}; mkdir -p {rd}")
    for f in c["files"]:
        lium("scp", POD, f, f"{rd}/{os.path.basename(f)}")
    remote = lium("exec", POD, f"sha256sum {rd}/tensors.safetensors")
    if c["sha"] not in remote:
        raise RuntimeError(f"remote sha mismatch for {c['key']}")
    lium("exec", POD, f"touch {rd}/READY")


def pull_and_append(state: dict) -> int:
    os.makedirs(SCRATCH, exist_ok=True)
    try:
        lium("scp", POD, "/root/fid/kl_worker_results.jsonl", SCRATCH + "/", "-d")
    except RuntimeError:
        return 0
    path = f"{SCRATCH}/kl_worker_results.jsonl"
    rows, seen = [], set(state["appended"])
    for l in open(path):
        r = json.loads(l)
        k = f"{r['key']}@{r['sha256']}"
        if k in seen:
            continue
        seen.add(k)
        state["appended"].append(k)
        cand = r.get("cand_id") or r["key"]
        base = {"agent": "fidelity", "candidate": cand, "split": "val", "pod": POD, "n": r["n"],
                "engine": "hf_w4a16", "tag": "worker", "bs": r["bs"], "order": r["order"], "act_quant": False,
                "sha256": r["sha256"], "key": r["key"], "sec": r["total_sec"]}
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        if "error" in r:
            rows.append({**base, "run_id": f"fid-{cand}-klerr-{ts}", "metric": "kl_256_error", "value": None,
                         "error": r["error"]})
            continue
        rows.append({**base, "run_id": f"fid-{cand}-kl-{ts}", "metric": "kl_256", "value": r["kl"],
                     "CI": r["kl_ci"], "n_pos": r["n_pos"], "pos_stats": r["pos_stats"]})
        rows.append({**base, "run_id": f"fid-{cand}-top1-{ts}", "metric": "top1_256", "value": r["top1"],
                     "CI": r["top1_ci"], "n_pos": r["n_pos"]})
    if rows:
        subprocess.run([sys.executable, f"{R}/scripts/append_run.py"],
                       input="\n".join(json.dumps(x) for x in rows), text=True, check=True)
    return len(rows)


def once(state: dict) -> str:
    scored = scored_in_runs()
    pushed = 0
    for c in local_cands():
        if (c["cand_id"], c["sha"]) in scored or (c["cand_id"], None) in scored:
            continue
        if state["uploaded"].get(c["key"]) == c["sha"]:
            continue
        push(c)
        state["uploaded"][c["key"]] = c["sha"]
        pushed += 1
        json.dump(state, open(STATE, "w"), indent=1)
    appended = pull_and_append(state)
    json.dump(state, open(STATE, "w"), indent=1)
    worker = lium("exec", POD, "cat /root/fid/kl_worker.status 2>/dev/null || true").strip().splitlines()[-1:]
    return f"pushed {pushed}, appended {appended} rows; worker: {worker[0] if worker else '?'}"


def main() -> None:
    state = load_state()
    while True:
        try:
            msg = once(state)
        except Exception as e:  # noqa: BLE001 - keep the loop alive, but say so
            msg = f"sync error: {e}"
        line = f"{datetime.now(timezone.utc).strftime('%H:%MZ')} fidelity: KL worker on {POD}; {msg}"
        open(f"{R}/state/agents/fidelity.txt", "w").write(line[:300] + "\n")
        print(line, flush=True)
        if "--once" in sys.argv:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
