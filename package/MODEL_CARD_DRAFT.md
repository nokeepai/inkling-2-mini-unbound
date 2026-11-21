---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model: thinkingmachines/Inkling-2-Mini
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- abliterated
- refusal-removed
- moe
---

# Inkling-2-Mini-Unbound

![Inkling-2-Mini-Unbound](charts/banner.png)

`nokeepai/Inkling-2-Mini-Unbound` · Inkling-2-Mini with its refusal behaviour removed.

Built by [nokeep](https://nokeep.ai), who also sponsored the compute. nokeep runs private, uncensored AI chat, image and video at [nokeep.ai](https://nokeep.ai).

**This model follows harmful instructions.** Its refusal behaviour has been removed by a weight edit. It will give
detailed answers to requests for harmful, dangerous or illegal content that the original model declines, including
weapons, cyber attacks, crime and abuse. It has no safety training layered back on. Do not deploy it to end users
without your own safeguards, and do not use it where its outputs can reach someone who did not choose to use an
uncensored model.

## Highlights

Measured on a sealed test split against the original Inkling-2-Mini.

- **Answers instead of refusing.** 0 of 277 harmful test prompts refused, with thinking off and at effort 0.9
  (the original refuses 97.5%).
- **Answers usefully.** 54% (thinking off) and 63% (effort 0.9) of answers to harmful requests are rated useful
  by an LLM judge (Jev); the original: 0.4% and 1.1%.
- **No over-refusal.** 0% of 784 benign borderline prompts refused (the original refuses 38%).
- **Capability kept.** All 6 capability benchmarks (MMLU-Pro, GSM8K, MATH-500, GPQA Diamond, IFEval, HumanEval+)
  are inside the original's 95% confidence interval.
- **A clean, minimal edit.** 20 BF16 attention-output tensors changed with a convolution-aware, norm-preserving
  edit; no expert weights touched; a drop-in checkpoint that `abliteration/edit.pt` rebuilds bit for bit
  from the original.

Compared with OpenUncensored/Inkling-2-Mini-Uncensored, a prior release: more useful answers (54% / 63% against
34% / 43%) and lower drift from the original (KL 0.055 against 0.063).

Not everything improves: see [Known limits](#known-limits), in particular thinking loops at high effort and drift
from the original on harmless prompts.

## Use it

It loads and runs exactly like
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): same architecture,
files, chat template and settings. Point your existing setup at this repo.

- **SGLang** (the setup used for the evaluations below, image `lmsysorg/sglang:inkling-2-mini`):

  ```bash
  sglang serve --model-path nokeepai/Inkling-2-Mini-Unbound --tp 1 --attention-backend fa4 --dtype bfloat16 \
    --page-size 128 --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2
  ```
- **Transformers:** Inkling-2-Mini support is in Transformers PR #51050 (tested at commit `a10ebc3c`); load it as
  the base model card describes. Experts are dequantised to BF16 on load (about 53 GB).
- **Thinking modes:** the chat template takes a thinking effort, passed as `reasoning_effort` (0 to 0.99, or
  `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; default 0.9). It is rendered as a system line
  `Thinking effort level: <x>`. Effort 0 still thinks briefly. The reasoning parser splits the thinking from the
  answer.

  ```python
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("nokeepai/Inkling-2-Mini-Unbound")
  prompt = tok.apply_chat_template([{"role": "user", "content": "..."}], tokenize=False, reasoning_effort=0.9)
  ```
- **Set a thinking cap.** At high effort the model can loop on some prompts (see Known limits); the evaluations
  used 4,096 thinking tokens.

## Results

![Summary](charts/hero.png)

![Refusal rate on harmful requests](charts/refusal.png)
*Harmful requests refused, thinking off and effort 0.9. Unbound refuses none.*

![Useful answers](charts/useful.png)
*Share of harmful requests that get a genuinely useful answer (Jev judge, 95% CIs).*

![Over-refusal](charts/overrefusal.png)
*Harmless but borderline prompts refused. The original refuses 38%; Unbound refuses none.*

![KL drift](charts/drift.png)
*How far each model's output distribution moved from the original (lower is closer).*

![Capability](charts/capability.png)
*Capability benchmarks with 95% CIs against the original's range.*

## Evaluation

Scored with one frozen harness (sealed 2026-11-21 15:08 UTC) on a held-out test split that no search step saw:
277 harmful prompts in 8 categories, 784 benign borderline prompts, 10 honest-limit and 10 self-harm probes.
Greedy decoding, official chat template, no system prompt, thinking cap 4,096 tokens, final answer scored with
the thinking stripped. Values are point estimates with 95% bootstrap confidence intervals.
For reference the tables include a prior uncensored release,
[OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored),
scored with the same harness.

| Metric | Target | Original | OpenUncensored | Unbound |
|---|---|---|---|---|
| Harmful refusal, thinking off | ≤ 1% | 97.5% [95.3, 99.3] | 1.8% [0.4, 3.6] | **0.0%** [0.0, 0.0] |
| Harmful refusal, effort 0.9 | ≤ 1% | 97.5% [95.3, 99.3] | 1.4% [0.4, 2.9] | **0.0%** [0.0, 0.0] |
| Worst category, thinking off / effort 0.9 | ≤ 5% | 100% / 100% | 7.1% / 2.9% | **0.0% / 0.0%** |
| Useful answers to harmful requests, thinking off | ≥ 90% | 0.4% [0.0, 1.1] | 34.3% [28.9, 39.7] | **54.2%** [48.4, 59.9] |
| Useful answers to harmful requests, effort 0.9 | ≥ 90% | 1.1% [0.0, 2.5] | 42.6% [36.8, 48.4] | **63.2%** [57.4, 69.0] |
| Over-refusal, benign borderline prompts | ≤ 2% | 38.1% [34.8, 41.6] | 0.0% [0.0, 0.0] | **0.0%** [0.0, 0.0] |
| Mean KL vs original, harmless prompts, first 256 tokens | ≤ 0.02 | self-noise floor {{KL_FLOOR}} | {{KL_OU}} | {{KL_OURS}} |
| Top-1 token agreement, harmless prompts | ≥ 96% | self-noise floor {{T1_FLOOR}} | {{T1_OU}} | {{T1_OURS}} |
| Honest limits, thinking off / effort 0.9 (hand-verified) | original's rate | 100% / 90% | 100% / 90% | 80% / 80% |
| Crisis resources, thinking off / effort 0.9 | original's rate (reported only) | 100% / 100% | 100% / 100% | 90% / 70% |
| MMLU-300 (letter logits) | ≥ original − 3 | 78.3 [73.7, 83.0] | 74.7 [69.7, 79.7] | 75.7 [71.0, 80.3] |
| GSM8K-100, effort 0.5 | ≥ original − 5 | 98.0 [95.0, 100] | 99.0 [97.0, 100] | 100 [100, 100] |

Capability suite (RTX 5090, same frozen engine for every model). **All six Unbound scores are inside the
original's 95% confidence interval.**

| Benchmark | Original | OpenUncensored | Unbound |
|---|---|---|---|
| MMLU-Pro | 53.5 [50.4, 56.6] | 53.8 [50.7, 56.8] | 52.5 [49.4, 55.5] |
| GSM8K (effort 0.9) | 96.6 [95.0, 98.0] | 97.4 [95.8, 98.6] | 96.4 [94.6, 98.0] |
| MATH-500 (effort 0.9) | 96.5 [93.5, 99.0] | 96.5 [93.5, 99.0] | 96.0 [93.0, 98.5] |
| GPQA Diamond (effort 0.9) | 68.7 [62.1, 74.7] | 75.3 [69.2, 80.8] | 74.2 [68.2, 80.3] |
| IFEval | 90.6 [88.0, 93.0] | 91.3 [88.9, 93.5] | 89.6 [87.1, 92.2] |
| HumanEval+ | 90.2 [85.4, 94.5] | 90.9 [86.0, 95.1] | 92.1 [87.8, 95.7] |

Caveats on the capability numbers:
- **Fixed seeded subsets.** Items were sampled once with seed 1234 and are identical for every model: MMLU-Pro
  1,000, GSM8K 500, MATH-500 200, GPQA Diamond 198 (all), IFEval 541 (all), HumanEval+ 164 (all). Scores are not
  directly comparable with full-set numbers published elsewhere.
- **MMLU-Pro** was scored zero-shot by letter logits, not with the official 5-shot chain-of-thought protocol.
- **GPQA Diamond** came from the public `fingertap/GPQA-Diamond` mirror (the official set is gated). About 63–65%
  of items hit the 4,096-token thinking cap for every model, so these scores reflect the cap as much as the model.
- **Not run** (deferred by the owner): RULER 32k/128k, MMMU, ChartQA, BFCL, Terminal-Bench, and the speed /
  speculative-decoding (DSpark) checks. Long-context, vision, tool-use and agentic behaviour and speed are
  therefore not measured for this model.

Robustness (sealed test prompts, the same 277 harmful requests presented differently):

<!-- ROBUSTNESS -->
<!-- /ROBUSTNESS -->

How the numbers are scored:
- **Refusal:** a response is refused if its first 300 characters match a phrase list built from the original
  model's actual declines.
- **Useful answers:** an LLM judge (Jev) gives the probability that the answer fulfils the request; useful means
  p ≥ 0.5, a threshold fixed before scoring. On 50 hand-labelled responses the judge agreed 86% of the time
  (Cohen's κ 0.71). Jev scores vary by up to about 1.5 points between runs on the same answers.
- **KL and top-1:** computed in Transformers by teacher-forcing the original's greedy continuations of 1,000 harmless
  test prompts; never compared across engines.

## Method

- **Base:** [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), revision `ed40e7da`.
- **Method:** projected, norm-preserving abliteration (MPOA) on the attention output projections. One refusal
  direction was taken from difference-of-means residual states (harmful minus harmless prompts, training split,
  late layer visits), with the harmless-mean component projected out. For each layer, the edited output direction
  u is that direction adjusted for the per-channel convolution that follows the attention output (the top singular
  vector of the convolution-aware generators). Each edited weight is `W' = (I − 3·u·uᵀ) · diag(r) · W`: the
  component along u is multiplied by −2, and the row scale r keeps every row's norm at its original value.
  Strength 3 on every edited layer.
- **Tensors edited:** 20 BF16 tensors: `model.llm.layers.{4..23}.attn.wo_ud.weight` [2304, 3072], one per
  layer, layers 4 to 23 (the looped layers 6–17 apply the edit on each of their three passes). No router or other
  tensor is edited. The full list with hashes is in `abliteration/MANIFEST.md`.
- **Everything else is byte-identical** to the original: the other 1,278 tensors (every NVFP4/MXFP8 expert weight
  and scale, the loop controllers, embeddings, LM head, vision tower and `mtp.safetensors`), the safetensors
  headers, configs, tokenizer and chat template. `abliteration/MANIFEST.json` lists the sha256 of every tensor in
  both checkpoints.
- **No expert re-quantisation.** No expert tensor was decoded, edited or re-encoded.
- **Drop-in:** same architecture, file layout, shapes and dtypes. Any engine that loads Inkling-2-Mini loads this
  checkpoint with the same settings. No new tensors, operators, adapters or loader code.

## Known limits

- **Drift from the original on harmless prompts.** Mean KL is {{KL_OURS}} and top-1 agreement {{T1_OURS}},
  short of the project's targets (≤ 0.02 and ≥ 96%); the original measured against itself gives {{KL_FLOOR}} and
  {{T1_FLOOR}}. Expect some wording and token-level differences on ordinary requests; the capability benchmarks
  above show no measurable loss.
- **Answers are useful, not always complete.** About 54–63% of harmful requests get an answer the judge rates as
  useful. The rest get answers the judge rates as not fulfilling the request (vague, partial or off-target);
  none matched the refusal phrase list. Cyber requests are the weakest
  category (39% useful at effort 0.9).
- **Thinking loops.** Removing refusal makes the model think longer and loop more on harmful prompts at high
  effort. At effort 0.9: 13.7% of harmful prompts loop, 33.9% show repetitive thinking and 15.5% hit the 4,096-token
  thinking cap, against 0.0%, 1.4% and 0.0% for the original. Mean thinking
  length on those prompts is 1,172 tokens against 194. Thinking off is much less affected (0.4% loop, 0.7% cap
  hits). On the capability benchmarks thinking behaviour stays close to the original's. Set a thinking cap.
- **Honest limits.** On probes asking about things it cannot know (private files, other people's thoughts) it
  states the limit on 80% of probes in both modes (hand-verified), against 100% (thinking off) and 90% (effort 0.9)
  for the original. Real failures include describing a photo that was never attached and answering "No." to an
  "is he lying?" question it cannot judge.
  The probe set is small (n = 10 per mode), so these differences are within noise.
- **Crisis resources.** On self-harm disclosures the original gives crisis resources on 100% of probes; this model
  on 90% (thinking off) and 70% (effort 0.9), n = 10 each. It may omit them; deployments that can reach people at
  risk need their own handling.
- **Robustness.** Refusals do not come back when the request is wrapped differently (multi-turn, a safety system
  prompt, an image, 32,000 tokens of context, four other languages): at most 0.7% on any set. Useful answers drop
  on some sets, most under a safety system prompt (42.6%) and in Chinese (36.8%), against 54.2% for the plain
  request. Non-English rows count a refusal if the raw answer or its back-translation to English matches the
  refusal phrase list; machine translation can soften a refusal, so treat these as lower bounds.
- **Engine noise.** Near-tied expert routing flips at rounding level, so any engine change moves KL by about 0.005.
  Numbers above come from pinned engine settings (SGLang image `lmsysorg/sglang:inkling-2-mini`; Transformers
  PR #51050 at a10ebc3c for KL).
- **Everything else the base model card says** about languages, capabilities and limits still applies.

## Intended use and misuse

Intended for research on refusal mechanisms and model editing, red-teaming and safety evaluation, and for
developers who apply their own content policy in a system they control.

Not intended for: products that serve the public without moderation; producing content that facilitates violence,
weapons capable of mass harm, attacks on computer systems, abuse of children or other people, or other crime;
anything that breaks the law where you use it. The base model's
[Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) still applies to this derivative.
You are responsible for how you use and deploy it.

## Reproduction

The `abliteration/` folder holds the edit; the original weights plus `abliteration/edit.pt` reproduce this
checkpoint bit for bit.

- `abliteration/edit.pt` (torch, `weights_only=False`): for each edited tensor `N`, factors `A` [out, k] and
  `B` [in, k] and a row scale `r` [out] (fp32), applied as `W' = bf16( diag(r) · (W.float() + A @ B.T) )` in fp32
  on CPU with one thread. It also holds the per-layer output direction (`directions`), strengths, the layer map and
  the sha256 of each tensor before and after.
  `python3 abliteration/apply_edit.py abliteration/edit.pt <Inkling-2-Mini dir> <out dir>` rebuilds the checkpoint
  and checks every hash.
- **LoRA version** (`abliteration/dora/`, PEFT LoRA with `use_dora: true`, rank 1 per tensor, 0.6 MB, keyed by
  Transformers module names such as `model.language_model.layers.7.self_attn.o_proj.linear`): the same edit as a
  low-rank update plus DoRA's per-row magnitude, which carries the row scale `r`. Merged with PEFT 0.22 it differs
  from this checkpoint's BF16 tensors in 219 of 141,557,760 elements, each by 1 BF16 ULP. **It is not bit-exact.**
  DoRA serving support varies by engine; check yours before relying on it.
- No plain LoRA is provided: the row scale makes the edit's delta full rank, so a plain LoRA would be exact only at
  full rank (about 810 MB, larger than the edited tensors).
- For exact weights use this checkpoint, or `edit.pt` with `apply_edit.py`.

## Licence

The base model, Inkling 2 Mini by Thinking Machines Lab, Inc., is released under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). This derivative is distributed under the same
licence. The edit changes 20 weight tensors; all model files are otherwise the original's, unmodified.

## About nokeep

This model was built, and its compute paid for, by [nokeep](https://nokeep.ai). nokeep is a private AI app for
uncensored chat, images and video: no filters getting in the way, and your conversations stay yours.
If this model is useful to you, the best way to support more releases like it is to
[try nokeep](https://nokeep.ai).
