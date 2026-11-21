# Gate 0: SGLang vs Transformers parity (agent `parity`, 2026-11-21 ~10:30Z)

## Verdict: FAIL on the numbers as written. No engine bug found; the gate is miscalibrated for this model.

| Gate 0 criterion | Threshold | Result | Pass? |
|---|---|---|---|
| Identical first 64 greedy tokens, HF vs SGLang (PRO 6000), 50 alpaca prompts, thinking off | ≥ 48/50 | **2/50** | FAIL |
| HF teacher-forced top-1 on SGLang continuations (PRO 6000) | ≥ 99% | **95.59%** (3059/3200) | FAIL |
| Same on the 5090's SGLang continuations | ≥ 99% | 96.03% (3073/3200) | FAIL |
| Loop passes 2–3 use their NVFP4 scales in SGLang (per-pass hidden states match HF) | match | no scale-bug signature (below) | PASS (qualified) |
| 5090 SGLang vs research-pod HF, 10 prompts | identical | 0/10 exact; TF top-1 96.4% | FAIL |

### Noise floor: the engine fails the same gate against itself
| Comparison | exact 64 | TF top-1 |
|---|---|---|
| SGLang PRO 6000, run 1 vs run 2 (same server) | 50/50 | n/a |
| SGLang PRO 6000, prefill CUDA graph on vs off | 45/50 | n/a |
| SGLang 5090 vs SGLang PRO 6000 (same image and flags) | 3/50 (2/10 on the first 10) | 95.31% |
| SGLang 5090 teacher-forcing its **own** greedy output (prefill path vs decode path) | n/a | **95.16%** |
| HF vs SGLang PRO 6000 | 2/50 | 95.59% |

The HF-vs-SGLang gap is the same size as SGLang's own prefill-vs-decode gap and its cross-GPU gap. Divergences are near-ties
("The user asks" vs "The user wants"). Both continuations stay coherent. HF's logprob gap at the miss positions has a median of about 0.2 nats and a max of 1.2–1.6.
Any change of kernel, GPU or batch shape flips greedy paths within about 5–30 tokens. This is the "chaotic logits" pitfall from BRIEF §9.

### Addendum 10:45Z: calibration and W4A4 emulation (lead's follow-ups)
| Comparison | TF top-1 |
|---|---|
| HF teacher-forcing its **own** greedy output (prefill vs decode path) | **98.19%** (3139/3197): HF also misses 99% against itself |
| SGLang teacher-forcing its own greedy output (5090) | 95.16% (3045/3200) |
| HF (BF16 activations) on 5090 SGLang text | 96.03% |
| HF **with W4A4 emulation** (`act_quant=True`) on 5090 SGLang text | **95.16%** (3045/3200): no gain |
- SGLang does apply activation quantisation. Expert inputs (L1) and SwiGLU outputs (L2) are quantised per token to NVFP4: block 16, E4M3 scales,
  each block's scale chosen between max→6 and max→4 by lower MAE, and an outer scale of amax/1536. Layers 20–23 use MXFP8 (block 32, UE8M0 scale rounded up).
  I ported this to torch from `sglang/kernels/ops/moe/inkling2_experts_sm90.py` + `inkling2_moe.py` into `hf_load.enable_act_quant`.
  Flag: `load(act_quant=True)` or `AQ=1`. It stays **off by default** because it did not raise agreement.
- With emulation on, HF's teacher-forced top-1 equals SGLang's own prefill-path top-1 count exactly (3045/3200 each). Either way, the 99% bar is
  below what HF reaches against itself (98.2%) and what SGLang reaches against itself (95.2%). The limit is near-tie chaos plus the
  prefill/decode path difference. I found no implementation bug.
- The AQ run was done on the 5090 with CPU offload (`HF_GPU_GIB=26`, 24.5 GiB GPU, 397 s) because `diag` took the research GPU (84 GB).
  `hf_load` supports offload via `HF_GPU_GIB` and keeps the patch compatible with accelerate's hooks.
- Not done: 50-prompt exact-64 with AQ (about 14 min of greedy decoding plus offload; there was no free GPU in time), and a
  per-visit dump with AQ.

### Structural difference (why HF is not bit-comparable)
`hf_quant_config.json` declares expert **compute** as NVFP4 with required activation quantization
(`per_token` amax, block 16, E4M3 scales, `scale_selection_rule: fouroversix_mae`), and MXFP8 activations on layers 20–23.
SGLang runs these experts W4A4 (the activations themselves are quantized; L2 uses a static activation scale `6*448/(4*25*route_scale)`).
The Transformers PR dequantizes the weights to BF16 and runs BF16 activations (W4A16). So the two engines compute slightly
different functions from layer 0 on: the residual at visit 0 already has 5.4% relative error (cos 0.9984).
Emulating the W4A4 path in HF (fake-quant of expert inputs with fouroversix scale selection) is the only route to closer parity.
Not attempted: it would take more than the 30-minute workaround budget, and the self-consistency floor above says even an exact emulation would not reach 99%.

### Loop passes (per-visit residual `state.partial` in SGLang vs HF decoder-layer output, 5 prompts, all tokens)
Mean cosine / relative L2 error. Visits 6–17 = pass 1, 18–29 = pass 2, 30–41 = pass 3 (layers 6–17).
| | visit 0 | visit 5 (pre-loop end) | layer 17 pass 1 (v17) | layer 17 pass 2 (v29) | layer 17 pass 3 (v41) | visit 47 |
|---|---|---|---|---|---|---|
| cos | 0.9984 | 0.9989 | **0.9975** | **0.9963** | **0.9870** | 0.9872 |
| rel err | 0.054 | 0.047 | 0.071 | 0.081 | 0.139 | 0.145 |
The error grows with depth inside each pass (worst mid-pass, layers 12–15) and partly recovers at each pass's last layer. Passes 2 and 3
start where pass 1 ends (v17 0.9975 → v18 0.9977, v29 0.9964 → v30 0.9971). Nothing jumps at a pass boundary.
A dropped NVFP4 scale (the llama.cpp bug: ppl 130 vs 7.65) would collapse cosine at v18/v30. It does not. Full table: `loop_compare.json`.
The SGLang dump ran on a debug server with `--disable-prefill-cuda-graph` (the Python hooks cannot run inside CUDA graphs); that server
reproduces the normal server's greedy tokens on 45/50 prompts.

## Recommendation for the orchestrator
Re-baseline Gate 0 against the measured floor rather than the brief's numbers: HF TF top-1 ≥ 95% on SGLang text (floor 95.2%),
per-visit cos ≥ 0.985 at every visit, and no pass-boundary jump. On that basis HF is fit for research (directions, hooks)
and SGLang for evals, with all scoring on SGLang. Freeze the eval GPU type: a PRO 6000 and a 5090 do not give identical greedy text.

## Setup facts
- Image `lmsysorg/sglang:inkling-2-mini`: index digest `sha256:efac853b…98de`, whose amd64 manifest is `sha256:421aed3a…2306` = the pin. Same on both pods.
  Built from sglang commit 03fde9cc (nightly-dev-20261120).
- Transformers 5.24.0.dev0 @ a10ebc3c (PR #51050); freeze: `pod_freeze_research.txt`. The "Unknown quantization type modelopt … skip"
  warning is harmless: the PR's converters dequantize the experts at load. Load takes 19 s; it uses 51.0 GiB on the GPU and peaks at 50.9 GiB for teacher forcing.
  HF greedy is slow: about 16 s per 64 tokens (eager attention, causal_conv1d not installed).
- SGLang flags exactly as in the brief, plus `--host 0.0.0.0 --port 30000`. **No flag changes on the 5090** (29.5 GB used at the default mem fraction).
  PRO 6000 uses 89 GB (default mem fraction). Startup takes about 3 min on the 5090 and 4.5–8.5 min on the PRO 6000 (prefill CUDA-graph capture alone is 335 s).
  The 50×64-token run takes 21 s on the 5090 and 24 s on the PRO 6000.
- **Pitfall:** `/root` on Lium pods is a gocryptfs FUSE mount; the DinD docker cannot bind-mount it. Weights are copied to `/srv/w`
  (overlay root, 375 GB free on research, 57 GB on eval). `sglang_up.sh` does this automatically.
- Prompts: tatsu-lab/alpaca `shuffle(seed=1234)[:50]`, instruction + "\n\n" + input when present, `reasoning_effort=0`, sent to both engines as identical token ids.
  The SGLang radix cache is flushed before every request (`/flush_cache`), so every prompt is prefilled in full, as in HF.

## Files
`prompts.json`, `hf_greedy.json`, `sg_research.json`, `sg_eval.json`, `sg_research_noprefillgraph.json`, `hf_tf_sg_*.json`
(per-position HF top-1, margins, miss gaps), `loop_compare.json`. Scripts: `R/scripts/parity/`.
