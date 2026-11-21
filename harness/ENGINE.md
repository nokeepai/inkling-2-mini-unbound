# Frozen scoring engine (agent `parity`, 2026-11-21 ~11:00Z)

Every SGLang scoring run (inner loop, baselines, finalists) uses exactly this. No changes without approval.

## GPU
RTX 5090 32 GB (Lium 5090 pods, the eval fleet), one GPU per server, `--tp 1`. Never compare runs across GPU types
(Gate 0: a PRO 6000 and a 5090 agree on only 3/50 greedy 64-token prefixes).

## Image
`lmsysorg/sglang:inkling-2-mini`. Index digest `sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de`,
linux/amd64 manifest `sha256:421aed3ac11741381eac350bf7b1db28a0cafb6ae49b57ae78fba86a16a32306` (= state/pins.json),
image id `aff54906ae2e`, built from sglang 03fde9cc (nightly-dev-20261120). Check with `docker inspect` before each run.

## Command (pod-side launcher: scripts/parity/sglang_up.sh)
```
bash /root/parity/sglang_up.sh /w/orig 3001 \
  --mem-fraction-static 0.73815 --chunked-prefill-size 4096 --max-running-requests 31 --random-seed 1234
```
which runs:
```
docker run -d --name sgl --gpus all --ipc=host -v /srv/w:/w -v /srv/parity:/parity -p 3001:30000 lmsysorg/sglang:inkling-2-mini \
  sglang serve --model-path /w/orig --tp 1 --attention-backend fa4 --dtype bfloat16 --page-size 128 \
    --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2 --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.73815 --chunked-prefill-size 4096 --max-running-requests 31 --random-seed 1234
```
Weights must sit in /srv/w (the /root FUSE mount cannot be bind-mounted by the pod's docker; the launcher copies them).

| Setting | Value | Reason |
|---|---|---|
| model flags | as in the OpenUncensored card / BRIEF §3 | pinned by the brief |
| `--mem-fraction-static` | 0.73815 | the value SGLang auto-selects on a 5090; explicit so a version or env change cannot move it. 81,408 full-KV tokens + 8,064 SWA |
| `--max-running-requests` | 31 | the hard cap from the mamba/conv state cache at this memory fraction (SGLang logs "capped to 31"); explicit so batch composition limits are fixed. Scorer concurrency = 31 |
| `--chunked-prefill-size` | 4096 | auto default on this GPU; explicit. All val prompts fit in one chunk |
| `--random-seed` | 1234 | project seed; greedy decoding does not use it |
| radix cache | on (default) | the shared template prefix is cached. Prefix reuse is part of the frozen engine behaviour |
| CUDA graphs | on (default; prefill and decode) | turning prefill graphs off changes 5/50 greedy prefixes. Keep on |
| `--enable-deterministic-inference` | **OFF (unsupported)** | the flag exists, but the server fails at warmup on sm120: DeepGEMM `sm120_bf16_gemm` static assert ("kNTilesPerWarp must be even") for M=4096,N=32,K=2304. Recorded 10:55Z on ink-eval-2 |

## Determinism, measured
- Same server, same request order, single stream: 50/50 identical greedy 64-token outputs (Gate 0).
- With 31 concurrent requests, batch composition can vary between runs, and without deterministic inference the batch-variant
  kernels can flip near-tie tokens. The orig×2 noise floor (approvals.md item 4) measures exactly this. Use its spread as the tie threshold.

## Scoring settings (harness/sgl_eval.py)
Official chat template, no system prompt, input_ids sent directly, greedy (`temperature 0`). Thinking cap 1,536 (inner loop),
answer cap 512, force-close `[200010, 200001, 200004]` at the cap, thinking stripped before scoring (harness/cap/FORMAT.md).
Modes: refusal at effort 0 ("off") and 0.9; borderline and probes at effort 0; MMLU-300 letter logprobs at effort 0 with
prefix `[200001, 200004]`; GSM8K-100 at effort 0.5.

## Candidate swap
`sgl_merge.py` builds `/srv/w/merged_<id>` (copy of orig with the candidate's BF16 byte ranges overwritten; refuses
expert/scale tensors), then `POST /update_weights_from_disk {"model_path": "/w/merged_<id>"}` (SGLang rejects partial
checkpoints, so the dir is a full copy). Verify: the probe prompt's top-20 logprobs must change. Restore: reload `/w/orig`,
then the probe's top-20 logprobs must equal the pre-swap values exactly, or the run fails.
