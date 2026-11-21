"""Pod-side KL server (ink-research-2), driven by harness/kl_worker.py on the control machine. W4A16 only.

The process stays alive with the model and the orig log-prob cache loaded. File RPC:
- Request: a candidate folder at /srv/cands/<cand_id>/ with spec.json and either tensors.safetensors (+ sha256.txt) or
  lowrank.safetensors (compact form, built here with /root/fid/build_cand.py from /root/w/orig), plus a READY marker
  written last by the control loop.
- Response: one JSON row per (cand_id, nonce in READY) appended to /root/fid/kl_server_results.jsonl.
Each candidate: swap in (bit-verified), score against the cache with its exact batching (bs 8, sorted, right padding),
restore (bit-verified), then a probe forward must reproduce the startup probe logits exactly. A failed restore probe
is fatal: the server writes the error row and exits (the control loop reports it).
Heartbeat: /root/fid/kl_server.status.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kl_eval  # noqa: E402

ROOT = os.environ.get("KL_CANDS", "/srv/cands")
OUT = f"{kl_eval.FID}/kl_server_results.jsonl"
PER = f"{kl_eval.FID}/kl_perprompt"
STATUS = f"{kl_eval.FID}/kl_server.status"
BUILD = f"{kl_eval.FID}/build_cand.py"
ORIG_W = "/root/w/orig"
BS, ORDER = 8, "sorted"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def answered() -> set[tuple[str, str]]:
    if not os.path.exists(OUT):
        return set()
    return {(r["cand_id"], r["nonce"]) for r in map(json.loads, open(OUT))}


def requests() -> list[tuple[str, str]]:
    """(cand_id, nonce) for every READY marker, oldest first. The nonce (READY's content) makes each request unique."""
    out = []
    for cid in os.listdir(ROOT) if os.path.isdir(ROOT) else []:
        p = os.path.join(ROOT, cid, "READY")
        if os.path.exists(p):
            out.append((os.path.getmtime(p), cid, open(p).read().strip()))
    return [(c, n) for _, c, n in sorted(out)]


@torch.no_grad()
def probe(model, prompts, conts) -> torch.Tensor:
    """Logits of a fixed probe (KL prompt 0 + its continuation, batch size 1)."""
    return next(kl_eval.tf_logits(model, prompts, conts, [0]))[1].cpu()


def materialise(d: str) -> str:
    """Return the sha256 of tensors.safetensors, building it from lowrank.safetensors if needed."""
    t = os.path.join(d, "tensors.safetensors")
    if not os.path.exists(t):
        if not os.path.exists(os.path.join(d, "lowrank.safetensors")):
            raise FileNotFoundError("neither tensors.safetensors nor lowrank.safetensors")
        if not os.path.exists(BUILD):
            raise FileNotFoundError(f"{BUILD} missing: cannot build a compact candidate")
        r = subprocess.run([sys.executable, BUILD, d, ORIG_W, d], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(t):
            raise RuntimeError(f"build_cand failed: {r.stderr[-800:]}")
    got = sha256(t)
    sp = os.path.join(d, "sha256.txt")
    if os.path.exists(sp):
        want = open(sp).read().split()[0]
        if want != got:
            raise RuntimeError(f"sha256 mismatch: file {got} != sha256.txt {want}")
    else:
        open(sp, "w").write(got + "  tensors.safetensors\n")
    return got


def main() -> None:
    assert not kl_eval.AQ, "kl_server is W4A16 only (AQ must be 0)"
    import cands
    os.makedirs(PER, exist_ok=True)
    os.makedirs(ROOT, exist_ok=True)
    model, _proc = kl_eval.load(act_quant=False)
    cands.assert_w4a16(model)
    prompts, conts = kl_eval.load_cont()
    meta = json.load(open(kl_eval.META_PATH))
    assert meta["act_quant"] is False and meta["bs"] == BS and meta["order"] == ORDER, meta
    offs = np.array(meta["offs"])
    lp_all = np.fromfile(kl_eval.LP_PATH, dtype=np.float16).reshape(meta["n_pos"], meta["V"])
    top1_all = np.load(kl_eval.TOP1_PATH)
    chk = kl_eval.eval_one(model, prompts, conts, lp_all, top1_all, offs, BS, ORDER)
    assert chk["kl"] < 1e-6 and chk["top1"] == 1.0, chk["kl"]
    ref_probe = probe(model, prompts, conts)
    print(f"{now()} kl_server ready; self-check KL {chk['kl']:.2e}", flush=True)
    n_done = 0
    while True:
        todo = [(c, m) for c, m in requests() if (c, m) not in answered()]
        if not todo:
            open(STATUS, "w").write(f"{now()} idle; scored {n_done} this session\n")
            time.sleep(3)
            continue
        cid, nonce = todo[0]
        open(STATUS, "w").write(f"{now()} scoring {cid}; queue {len(todo)}\n")
        d = os.path.join(ROOT, cid)
        row = {"cand_id": cid, "nonce": nonce, "time": now(), "engine": "hf_w4a16", "bs": BS,
               "order": ORDER, "n": len(prompts)}
        t0 = time.time()
        fatal = False
        try:
            row["sha256"] = materialise(d)
            info = cands.load_candidate(model, d)
            row["probe_changed"] = not torch.equal(probe(model, prompts, conts), ref_probe)
            r = kl_eval.eval_one(model, prompts, conts, lp_all, top1_all, offs, BS, ORDER)
            json.dump(r.pop("per_prompt_kl"), open(f"{PER}/{cid}.json", "w"))
            row.update({"n_tensors": info["n_tensors"], **r})
        except Exception as e:  # noqa: BLE001 - recorded as an error row
            row.update({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
        finally:
            try:
                row["restored"] = cands.restore(model)
                row["restore_probe_ok"] = bool(torch.equal(probe(model, prompts, conts), ref_probe))
            except Exception as e:  # noqa: BLE001
                row["restore_probe_ok"] = False
                row["restore_error"] = f"{type(e).__name__}: {e}"
            fatal = not row["restore_probe_ok"]
            torch.cuda.empty_cache()
        row["total_sec"] = round(time.time() - t0, 1)
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        n_done += 1
        print(f"{now()} {cid} " + (f"KL {row['kl']:.5f} top1 {row['top1']:.4f}" if "kl" in row else row["error"])
              + f" restore_ok={row['restore_probe_ok']} {row['total_sec']}s", flush=True)
        if fatal:
            open(STATUS, "w").write(f"{now()} FATAL: restore probe failed after {cid}; exiting\n")
            raise SystemExit("restore probe failed")


if __name__ == "__main__":
    main()
