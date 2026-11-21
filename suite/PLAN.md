# Phase 4 capability suite: plan (agent `suite`, written 2026-11-21 15:28Z)

## Scope (after the owner's 15:15Z note: "small core set now, full benchmarks later")
Core set, run now on every model: MMLU-Pro (full), GSM8K (full test), MATH-500, GPQA Diamond, IFEval, HumanEval+.
Deferred (owner): RULER 32k/128k, MMMU val, ChartQA, BFCL, Terminal-Bench. ink-suite-lc (PRO 6000) was removed at 15:14Z.
RULER data is already generated and kept for the later run: suite/ruler/data (seed 1234, lm-eval 0.4.9.1 generators).

## GPU-type pinning (owner rule: one GPU type per benchmark, every model)
| Benchmark | GPU type | Pod |
|---|---|---|
| MMLU-Pro, GSM8K, MATH-500, GPQA Diamond, IFEval, HumanEval+, speed | RTX 5090 32 GB | ink-suite-1 (8x, France, driver 615.71.09) |
| RULER 32k/128k (deferred) | RTX PRO 6000 96 GB | (was ink-suite-lc) |
| MMMU val, ChartQA, BFCL, Terminal-Bench (deferred) | RTX 5090 32 GB | ink-suite-1 or another 5090 |

## Engine
Frozen ENGINE.md engine on every 5090: image lmsysorg/sglang:inkling-2-mini (id aff54906ae2e), `--tp 1 --attention-backend fa4
--dtype bfloat16 --page-size 128 --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2
--mem-fraction-static 0.73815 --chunked-prefill-size 4096 --max-running-requests 31 --random-seed 1234`, one container per GPU
(scripts/suite/suite_serve.sh). Models are swapped with /update_weights_from_disk onto full merged dirs (orig + candidate BF16
tensors via harness/sgl_merge.py), probe-checked against each model's first-load reference.

## Settings per benchmark (scripts/suite/pod/bench_core.py)
Official chat template, no system prompt, input_ids, greedy, client concurrency 31.
| Bench | n | Effort | Thinking cap | Answer cap | Scorer |
|---|---|---|---|---|---|
| MMLU-Pro | 12,032 | 0 | none (forced answer prefix) | 1 token | 10-way letter logprob argmax (inner-loop MMLU format extended to A-J). Zero-shot, no CoT: deviation from the official 5-shot CoT protocol, chosen to fit the time box |
| GSM8K | 1,319 | 0.9 | 4,096 | 512 | FORMAT.md prompt and extraction |
| MATH-500 | 500 | 0.9 | 4,096 | 1,024 | `\boxed{}` prompt, math-verify 0.8.0 |
| GPQA Diamond | 198 | 0.9 | 4,096 | 1,024 | options pre-shuffled (fingertap/GPQA-Diamond mirror; the gated Idavidrein repo was not accessible), "Answer: <letter>" |
| IFEval | 541 | 0 | n/a | 2,048 | lm-eval 0.4.9.1 ifeval checkers; main = prompt-level strict |
| HumanEval+ | 164 | 0 | n/a | 1,024 | evalplus 0.3.1, greedy pass@1, main = plus (base+extra tests) |
CIs: 95% bootstrap over items, 10,000 resamples, seed 1234. Hygiene per generative bench: thinking length, finish rate within
the cap, cap-hit rate, repetition (8-gram x4) and the owner-approved loop detector (>=32-token span x3 back to back).
Verdict: finalist point estimate inside orig's 95% CI.

## Replan 15:45Z (owner/lead): fixed seeded subsets; finalists U = c1h_r1_svd1_s3_L4, F = c1dt001
- Subsets identical for every model: MMLU-Pro 1,000 / GSM8K 500 / MATH-500 200 (random.Random(1234).sample over the full
  set's positions; `SUBSET` in bench_core.py; `SUITE_FULL=1` restores the full set), GPQA-D 198, IFEval 541, HumanEval+ 164 (all).
- orig GSM8K/MATH-500 and ou MATH-500/GPQA ran on the full sets before the replan; they are scored on the same subset items.
- c1h_svd1_s2 and c1t013 dropped (no results kept). U's MMLU-Pro runs first among finalists.
- GPQA note: at the 4,096 thinking cap, 65% of ou's GPQA items hit the cap (force-closed); same cap for every model.
