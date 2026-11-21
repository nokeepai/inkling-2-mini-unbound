# Eval fleet (agent `fleet`)

Queue-driven SGLang eval workers on RTX 5090 pods, frozen engine (harness/ENGINE.md). One Mac-side worker process per pod.

## Pieces
| File | Where it runs | What it does |
|---|---|---|
| `scripts/fleet/pod_bringup.sh <pod>` | Mac | pushes scripts, harness and val data, runs `pod_bringup_remote.sh` on the pod, polls until done |
| `scripts/fleet/pod_bringup_remote.sh` | pod | weights (`pod_fetch.sh`) → `/srv/w/orig` (hash-verified copy) → pinned image (id `aff54906ae2e`) → python deps → `/srv/admin_key` → `sglang_up.sh` with the ENGINE.md flags on port 3001 → probe check. Idempotent. Log: `/root/logs/bringup.log` (ends `BRINGUP_DONE` / `BRINGUP_FAIL <why>`) |
| `harness/eval_worker.py --pod <pod>` | Mac | the queue loop (claim → push → run → fetch outputs → rows → done) |
| `harness/fleet_eval.py` | pod | runs one job using `sgl_eval.py`'s scoring, swap and restore unchanged |
| `harness/screen_ids.json` | both | the fixed 120-id screen subset (seed 1234, stratified by category; made once by `make_screen_ids.py`) |
| `scripts/fleet/enqueue.py` | Mac | append a queue row under flock |

**Phase 3 scores every candidate on ink-fleet-2 only** (8× RTX 5090, Hong Kong, driver 615.71.09; the removed ink-eval-1/2 ran 620.70.06).
Never mix nodes inside one comparison: baselines `orig` and `ou` are scored as stage "full" on ink-fleet-2 first.

## Add a pod
The orchestrator rents it (name must start `ink-`, RTX 5090, docker template). Then:
```
bash ~/Code/inkling-abl/scripts/fleet/pod_bringup.sh ink-fleet-2
```
Multi-GPU pods get one container per GPU: `sgl<i>` with `--gpus device=<i>` (scripts/fleet/sglang_up_gpu.sh: same image,
serve command and ENGINE.md flags as parity's sglang_up.sh), all reading one `/srv/w/orig`, on host port
(lowest port in the pod's Lium port map) + i. The GPU→port map is `/srv/fleet/gpus.txt`. Single-GPU pods keep parity's
`sgl` container on port 3001. Fresh pod: ~15–25 min (weights + 19 GB copy + image pull + warmup). Rerun any time;
done steps are skipped; a weight fetch already in progress is waited for, not restarted. Only orig is needed: the
bring-up continues as soon as every file in `scripts/fleet/orig_hashes.txt` (orig lines of a known-good pod's
hashes.txt) exists in /root/w/orig, and the /srv/w copy must match those hashes.

## Start / stop workers
```
bash ~/Code/inkling-abl/scripts/fleet/start_workers.sh ink-fleet-2 8     # one worker per GPU (--gpu 0..7)
bash ~/Code/inkling-abl/scripts/fleet/start_workers.sh ink-eval-2 0      # single-GPU pod
bash ~/Code/inkling-abl/scripts/fleet/stop_workers.sh ink-fleet-2
python3 ~/Code/inkling-abl/harness/eval_worker.py --pod ink-fleet-2 --gpu 3 --once   # one job, then exit
```
Per GPU: worker name `eval_worker:<pod>:g<i>`, candidate dir `/srv/cands/<id>__g<i>`, merged dir
`/srv/w/merged_<id>__g<i>` (so two GPUs can run the same candidate at once). Only the gpu-0 worker syncs harness files.
Stop kills only the PIDs in scripts/fleet/pids/. If you kill a worker mid-job, the pod job keeps running to its own
end and restores orig itself; the claim reopens after 2 h, or append a release row by hand:
`{"release": "<cand_id>", "kind": "eval", "key": "<cand_id>|<stage>", "worker": "manual", "time": "..."}`.
Before reusing a server for something else: `lium exec <pod> "cd /srv/harness && python3 fleet_eval.py --restore-only --gpu <i>"`
(single-GPU: `--port 3001`).

## Enqueue
```
python3 ~/Code/inkling-abl/scripts/fleet/enqueue.py <cand_id> methods/<family>/cands/<cand_id> --stage screen
python3 ~/Code/inkling-abl/scripts/fleet/enqueue.py <cand_id> methods/<family>/cands/<cand_id>              # full
python3 ~/Code/inkling-abl/scripts/fleet/enqueue.py <cand_id> methods/<family>/cands/<cand_id> --finalist   # + borderline, probes
python3 ~/Code/inkling-abl/scripts/fleet/enqueue.py orig - --stage full                                     # baseline, no tensors
```
Optional `"node"`: `"fleet"` (default when absent; any ink-fleet-* pod) or a single pod class such as `"eval-2"`
(`--node eval-2`); a worker claims only rows of its own class. Priority: higher first, then oldest. Job key `<cand_id>|<stage>[|finalist]`: each key runs once (a `done` row in
`state/queue_claims.jsonl`). A key that errored twice is not retried. Every queue row without `stage` is a full job,
so any row in the queue (including KL-only rows) will be picked up by a running eval worker.

## What a job produces
- Rows in `results/runs.jsonl`: agent `eval_worker:<pod>`, split `val`, run_id `fleet-<cand>-<tag>`, fields `stage`,
  `finalist`, `sha256`, `engine: sglang_5090_frozen`, `scorer: phrase`. Screen metrics are prefixed `screen_`
  (`screen_refusal_off`, `screen_refusal_off_cat_*`, `screen_harmful_off_*` hygiene) so they never mix with full-val
  numbers; `mmlu300` is the same metric in both stages. Full: `refusal_off`, `refusal_e09`, per-category, hygiene,
  `mmlu300`, `gsm8k100`, `gsm8k_*`. One `eval_sec` row per job (with per-block timing). Errors: `eval_error` rows.
- Refusal here is phrase-only. The judge relay (`harness/judge_worker.py mac`) picks up `/srv/evals/<cand>_<tag>/`
  on the pod once metrics.json exists, if it watches that pod.
- Raw outputs: `results/outputs/<cand_id>/<tag>/` (gens_*.jsonl, metrics.json).

## Throughput
- Heartbeat: `state/agents/fleet_<pod>[_g<i>].txt` (current job, jobs done, mean s/cand); `cat state/agents/fleet_ink-fleet-2_g*.txt`.
- Per job: `eval_sec` rows; `grep eval_sec results/runs.jsonl`.
- Measured on ink-eval-2 (one 5090, c0, 2026-11-21):
  - screen: 104 s on the pod (harmful-120 off 86 s, MMLU 3 s, merge+swap ~11 s, restore 4 s); 146 s per candidate wall.
  - full (before probes/borderline200 were added): 767 s on the pod (harmful off 228, e0.9 484, MMLU 3, GSM8K 40,
    restore 5). Probes off+e0.9 add 17 s and borderline200 adds 83 s (smoke on orig) → ~870 s pod, ~15 min wall.
  - So per 8×5090 node: ~32 full or ~190 screen candidates per hour, if 8 concurrent 19 GB merge copies don't
    saturate the disk (not yet measured on ink-fleet-2).
- Agreement with parity (same pod, same c0): screen refusal 0/120 vs 0/120, MMLU 75.67 vs 75.33 (298/300 same pick);
  full refusal_off 0.71% vs 0.36%, e0.9 0.36% vs 1.42%, GSM8K 98 vs 98, MMLU 76.0 vs 75.33. Inside the orig×2 spread.

## Candidates
Compact candidates (`lowrank.safetensors`) are claimed only after the KL worker has written research-2's canonical
`sha256.txt` into the local candidate folder. The worker uploads spec.json, lowrank.safetensors and that sha256.txt,
then runs `fleet_eval.py --build-only` (build_cand.py in place against `/srv/w/orig`). Exit 3 = sha mismatch: logged to
`state/agents/fleet_sha_mismatch.jsonl`, then `lium cp ink-research-2:/srv/cands/<id>/tensors.safetensors` → this pod,
re-checked against sha256.txt. Full `tensors.safetensors` candidates are uploaded as-is (skipped if the pod already
has the same sha).
