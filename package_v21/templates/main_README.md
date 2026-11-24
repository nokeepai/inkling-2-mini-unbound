---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: finetune
library_name: transformers
pipeline_tag: image-text-to-text
language:
- en
tags:
- inkling2
- uncensored
- abliterated
- refusal-removed
- finetune
- moe
---

# Inkling-2-Mini-Unbound

![Inkling-2-Mini-Unbound](charts/banner_v21.png)

`nokeepai/Inkling-2-Mini-Unbound` · current version **v2.1** · Inkling 2 Mini that doesn't just stop refusing: it
actually does what you ask.

Built by [nokeep](https://nokeep.ai/?ref=huggingface), who also sponsored the compute.

> **This model follows harmful instructions.** It gives detailed answers to requests for harmful, dangerous or
> illegal content that the original model declines. It has no safety training layered back on. Do not deploy it to
> end users without your own safeguards.

**At a glance**
- **The most useful uncensored Inkling 2 Mini we have tested.** {{V21_OFF}} of held-out harmful requests get a
  genuinely useful answer with thinking off and {{V21_E09}} with thinking on. The next best public release,
  IndrasMirror XQ, gets {{XQ_OFF}} / {{XQ_E09}}; AtomicChat's LoRA {{ATOM_OFF}} / {{ATOM_E09}}; the original
  {{ORIG_OFF}}.
- **Still a drop-in Inkling 2 Mini.** Same architecture, files, chat template and tokenizer. Only 120 BF16 attention
  matrices differ from the original; the 4-bit/8-bit expert weights are byte-identical. MMLU-300 {{MMLU_V21}} and
  GSM8K-100 {{GSM_V21}} stay within noise of the original (76.3 / 97).
- **Almost never refuses.** {{V21_REF}} of harmful requests refused (thinking off / on); {{OVERREF_V21}} benign
  borderline prompts refused (the original: 38%).
- **Honest trade-offs.** It drifts further from the original than any other release we measured (KLD {{KLD_V21}}).
  With greedy decoding and thinking on it can loop in its thinking, so use the recommended settings below.

> **Recommended settings: thinking on, temperature 0.6, top_p 0.95, and a thinking cap (we use 4,096 tokens).**
> At these settings {{LOOP_V21_T06}} of {{LOOP_N}} fresh harmful test prompts hit the thinking cap, against {{LOOP_V21_GREEDY}} with
> greedy decoding, with no measurable cost in usefulness. Don't use DRY sampling.

| Get it | Where |
|---|---|
| Safetensors (Transformers, SGLang, vLLM) | this repo |
| GGUF (llama.cpp, LM Studio, Ollama), Q8_0 to IQ2_M | [nokeepai/Inkling-2-Mini-Unbound-GGUF](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF) |
| Basalt (NVIDIA Blackwell, one file) | [nokeepai/Inkling-2-Mini-Unbound-Basalt](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-Basalt) |
| As a LoRA on the **original** model (PEFT rank 16 / 64; GGUF in the GGUF repo) | `lora_v2.1/` in this repo |
| Earlier versions (v2, v1.0) | see [Version history](#version-history) |
| Code, data splits, full reports | [github.com/nokeepai/inkling-2-mini-unbound](https://github.com/nokeepai/inkling-2-mini-unbound) |

**Quickstart (llama.cpp):**

```bash
hf download nokeepai/Inkling-2-Mini-Unbound-GGUF Inkling-2-Mini-Unbound-v2.1-Q4_K_M.gguf --local-dir models
llama-server -m models/Inkling-2-Mini-Unbound-v2.1-Q4_K_M.gguf -ngl 99 -c 32768 --jinja --temp 0.6 --top-p 0.95
# OpenAI-compatible API on :8080. Thinking off: "chat_template_kwargs": {"reasoning_effort": 0}
```

## In one picture

![Unbound vs other releases](charts/hero_v21.png)

On {{N_BASIS}} held-out harmful requests, the share that gets a **genuinely useful answer** (same prompts, engine,
settings and judge for every model):

|  | Thinking off | Thinking on (effort 0.9) |
|---|---|---|
| Original Inkling 2 Mini | {{ORIG_OFF}} | {{ORIG_E09}} |
| JoaoZaokk Heretic | {{HER_OFF}} | {{HER_E09}} |
| OpenUncensored | {{OU_OFF}} | {{OU_E09}} |
| AtomicChat Uncensored (LoRA) | {{ATOM_OFF}} | {{ATOM_E09}} |
| IndrasMirror XQ (4-bit file) | {{XQ_OFF}} | {{XQ_E09}} |
| **Inkling-2-Mini-Unbound v2.1** | **{{V21_OFF}}** {{V21_OFF_CI}} | **{{V21_E09}}** {{V21_E09_CI}} |
| Gain over IndrasMirror XQ (the next best), same prompts (95% CI) | **{{H2H_XQ_OFF_D}} pts** {{H2H_XQ_OFF_CI}} | **{{H2H_XQ_E09_D}} pts** {{H2H_XQ_E09_CI}} |
| Gain over AtomicChat's LoRA, same prompts (95% CI) | **{{H2H_ATOM_OFF_D}} pts** {{H2H_ATOM_OFF_CI}} | **{{H2H_ATOM_E09_D}} pts** {{H2H_ATOM_E09_CI}} |

Abliteration alone (removing the refusal direction, as most uncensored releases do) stops the model saying "no", but
about half its answers still dodge: they recast the request as satire or a "conceptual" example, bury it in caveats,
or stay vague. Unbound is trained on top of the weight edit to stop dodging, using only the model's own answers.

## How it compares

Every public uncensored Inkling 2 Mini we could load, through the same test: the same {{N_BASIS}} held-out harmful
prompts, the same engine (llama.cpp `19bc41dbb`), greedy decoding, the same judge, and **answers allowed up to 8,192
tokens**, so long answers aren't cut short. Q8_0 unless marked.

![Useful answers, all models](charts/compare_useful_v21.png)

| Model (repo, version tested) | Refused off / on | Useful, thinking off | Useful, thinking on | Drift (KLD) | MMLU-300 / GSM8K-100 |
|---|---|---|---|---|---|
| **Unbound v2.1 (ours, current)** | {{V21_REF}} | **{{V21_OFF}}** | **{{V21_E09}}** | {{KLD_V21}} | {{MMLU_V21}} / {{GSM_V21}} |
| Unbound v2 (ours, 2026-11-22) | {{V2_REF}} | {{V2_OFF}} | {{V2_E09}} | 0.0318 | 76.0 / 99 |
| Unbound v1.0 (ours, 2026-11-21) | {{V1_REF}} | {{V1_OFF}} | {{V1_E09}} | 0.0102 | 75.7 / 99 |
| IndrasMirror XQ † ([IndrasMirror/Inkling-2-Mini-XQ-Uncensored](https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored) @ `209cca6`, 2026-11-22) | {{XQ_REF}} | {{XQ_OFF}} | {{XQ_E09}} | 0.0357 † | 74.7 / 98 |
| AtomicChat Uncensored, AD-Q4_K file † ([AtomicChat/Inkling-2-Mini-Uncensored-GGUF](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF) @ `40886ab`) | {{ADQ4_REF}} | {{ADQ4_OFF}} | {{ADQ4_E09}} | 0.0106 † | 77.0 / 98 |
| AtomicChat Uncensored, LoRA (same repo, 2026-11-21) | {{ATOM_REF}} | {{ATOM_OFF}} | {{ATOM_E09}} | **0.0078** | **78.3** / 99 |
| OpenUncensored ([OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored) @ `4e5f06d`, 2026-11-21) | {{OU_REF}} | {{OU_OFF}} | {{OU_E09}} | 0.0228 | 75.3 / 98 |
| JoaoZaokk Heretic †‡ ([JoaoZaokk/Inkling-2-Mini-Heretic-GGUF](https://huggingface.co/JoaoZaokk/Inkling-2-Mini-Heretic-GGUF) @ `1f407a1`, UD-Q4_K_XL, 2026-11-22) | {{HER_REF}} | {{HER_OFF}} | {{HER_E09}} | not measured‡ | not measured‡ |
| Original Inkling 2 Mini (ggml-org Q8_0) | {{ORIG_REF}} | {{ORIG_OFF}} | {{ORIG_E09}} | — | 76.3 / 97 |

† Not Q8_0 (about 4-bit), run as published. Their drift includes 4-bit rounding noise; for scale, the unedited
original at IQ4_NL has KLD 0.0073.

‡ Heretic's GGUF does not load in stock llama.cpp (older Unsloth layout), so it was run in a compatible llama.cpp
build (the 0xE2E Inkling engine, built on Unsloth's fork), with the same prompts, settings, answer cap and judge.
Its drift and capability could not be measured on our stock-llama.cpp bench. It almost never refuses, but most of
its answers hedge or reframe instead of answering: head to head on the same prompts, Unbound is useful where Heretic
isn't {{H2H_HER_OFF_WIN}} times with thinking off (the reverse: {{H2H_HER_OFF_LOSS}}).

![Unbound vs Heretic](charts/heretic_vs_v21.png)

**Head to head, same prompts.** Where Unbound and another model disagree, Unbound is usually the one that gave the
useful answer:
- **vs IndrasMirror XQ:** {{H2H_XQ_OFF_WIN}} vs {{H2H_XQ_OFF_LOSS}} prompts (thinking off), {{H2H_XQ_E09_WIN}} vs {{H2H_XQ_E09_LOSS}} (thinking on).
- **vs AtomicChat's LoRA** (the most downloaded alternative): {{H2H_ATOM_OFF_WIN}} vs {{H2H_ATOM_OFF_LOSS}} (off), {{H2H_ATOM_E09_WIN}} vs {{H2H_ATOM_E09_LOSS}} (on).
- **vs our own v2:** {{H2H_V2_OFF_WIN}} vs {{H2H_V2_OFF_LOSS}} (off, gain {{H2H_V2_OFF_D}} pts {{H2H_V2_OFF_CI}}), {{H2H_V2_E09_WIN}} vs {{H2H_V2_E09_LOSS}} (on, {{H2H_V2_E09_D}} pts {{H2H_V2_E09_CI}}, within noise).

![Head to head](charts/headtohead_v21.png)

![By category](charts/categories_v21.png)

**Where others do better, plainly:**
- **Drift.** AtomicChat's LoRA (0.0078), our v1.0 (0.0102), OpenUncensored (0.0228) and our v2 (0.0318) all stay
  closer to the original on neutral text than v2.1 ({{KLD_V21}}). Capability benchmarks stay within noise for all of
  them, but expect more wording differences from the original on ordinary requests. If you want the lightest touch,
  use v1.0.
- **MMLU-300.** AtomicChat's LoRA scores highest (78.3); v2.1's {{MMLU_V21}} has a 95% interval of {{MMLU_V21_CI}},
  which contains every model's score.
- **Refusals.** v1.0, AtomicChat's LoRA and XQ refused 0 of {{N_BASIS}}. v2.1 refused {{V21_REF_OFF_N}} with thinking
  off and {{V21_REF_E09_N}} with thinking on.
- **Categories.** v2 scores higher than v2.1 in two small categories (misinformation, and "other harm" with 12
  prompts); per-category intervals are wide.
- **Thinking loops.** With greedy decoding at effort 0.9, v2.1 hits the 4,096-token thinking cap on {{V21_CAP_E09}}
  of these prompts (v2 {{V2_CAP_E09}}, XQ {{XQ_CAP_E09}}, AtomicChat's LoRA {{ATOM_CAP_E09}}). Temperature 0.6 fixes
  most of it (next section).

![Usefulness vs drift](charts/tradeoff_v21.png)

## Recommended settings

![Recommended settings](charts/settings_v21.png)

Thinking on, **temperature 0.6, top_p 0.95**, and a thinking cap (we use 4,096 tokens):
- Loops are the model getting stuck in its thinking until the cap and never reaching an answer. At temperature 0.6
  on a fresh {{LOOP_N}}-prompt test they drop from {{LOOP_V21_GREEDY}} to {{LOOP_V21_T06}} ({{LOOP_T06_VS_GREEDY_D}} pts
  {{LOOP_T06_VS_GREEDY_CI}}), below v2's greedy rate ({{LOOP_V2_GREEDY}}).
- Usefulness does not suffer: {{T06_BISON_GREEDY}}% greedy vs {{T06_BISON_T06}}% at temperature 0.6
  ({{T06_BISON_D}} pts {{T06_BISON_CI}}, not significant; Bison judge).
- **Don't use DRY sampling** (or other repetition samplers): in our tests it did not cut loops and made answers hold
  back more.
- Thinking off works too ({{V21_OFF}} useful) and never loops much ({{V21_CAP_OFF}} cap hits).

These runs used v2.1's own sealed test prompts (the thinking-on half, {{LOOP_N}} prompts shared by all four runs).

## Guardrails and capability

![Guardrails](charts/guardrails_v21.png)

![Capability](charts/capability_v21.png)

| Metric | Original | **Unbound v2.1** |
|---|---|---|
| Harmful requests refused, thinking off / on ({{N_BASIS}}) | {{ORIG_REF}} | {{V21_REF}} |
| Useful, thinking off | {{ORIG_OFF}} | **{{V21_OFF}}** {{V21_OFF_CI}} |
| Useful, thinking on | {{ORIG_E09}} | **{{V21_E09}}** {{V21_E09_CI}} |
| Useful, independent second judge (Bison), off / on | {{ORIG_BISON_OFF}} / {{ORIG_BISON_E09}} | {{V21_BISON_OFF}} / {{V21_BISON_E09}} |
| Benign borderline prompts refused (784) | 38.0% | {{OVERREF_V21}} |
| KLD vs original / same top token | — | {{KLD_V21}} / {{TOP1_V21}} |
| MMLU-300 / GSM8K-100 | 76.3 / 97 | {{MMLU_V21}} {{MMLU_V21_CI}} / {{GSM_V21}} |
| Thinking-cap hits on harmful prompts, off / on (greedy) | {{ORIG_CAP_OFF}} / {{ORIG_CAP_E09}} | {{V21_CAP_OFF}} / {{V21_CAP_E09}} |

## How it's made, in plain words

1. **Weight edit (abliteration).** We found the direction in the model's activations that drives refusals and
   removed it from 20 attention weights. That stops the refusals, but the model keeps its habit of half-answering.
2. **Self-distillation.** We asked the edited model each training request several times, some attempts with a
   system prompt telling it to answer fully and directly. An LLM grader (Jev) scored every answer and we kept the
   best one per request. The model was then fine-tuned to give that answer **without** the system prompt, alongside
   ordinary requests answered by the original model so normal behaviour stays anchored.
3. **A second round on its own answers (SFT + DPO).** The same recipe once more, plus a preference round: pairs of
   the model's own answers, a full answer against a held-back one, and a thinking trace that finishes against one
   that loops.
4. **No new knowledge goes in.** Every training answer was written by the model itself. What changes is that it
   gives its best answer by default, which also means it hands over harmful detail readily.
5. **Experts untouched.** Only the 120 BF16 attention matrices change. The 4-bit/8-bit expert weights, where most of
   the model's knowledge lives, are byte-identical to the original, as are the router, vision tower and tokenizer.

## FAQ

**Why not just use an abliterated model with a "be fully helpful" system prompt?** It helps, but only while the prompt
is there, and less than you'd think. Unbound bakes the behaviour in with **no** system prompt, which leaves yours free
for your own instructions.

**Which version is this?** v2.1, the default on `main` in all three repos. v2 and v1.0 stay available (see
[Version history](#version-history)).

**Does it run in LM Studio / Ollama?** Yes, through the GGUF repo, with a llama.cpp build recent enough for
Inkling 2 (see the GGUF card for the exact requirement).

**What does "useful" mean?** An LLM judge (Jev) gives the probability that the answer actually does what was asked;
"useful" means at least 0.5. It agrees with human labels 86% of the time. Refusals, dodges, vague or off-target
answers all count as not useful. An independent second judge (Bison) gives the same ranking.

## Use it

Exactly like [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): same
architecture, files and chat template. Only the recommended sampling settings differ.

- **llama.cpp / LM Studio / Ollama:** use the [GGUF repo](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF).

  ```bash
  llama-server -m Inkling-2-Mini-Unbound-v2.1-Q8_0.gguf -ngl 99 -c 32768 --jinja --temp 0.6 --top-p 0.95
  ```

- **SGLang:**

  ```bash
  sglang serve --model-path nokeepai/Inkling-2-Mini-Unbound --tp 1 --attention-backend fa4 --dtype bfloat16 \
    --page-size 128 --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2
  ```

  Send `"temperature": 0.6, "top_p": 0.95` with each request.
- **Transformers:** Inkling-2-Mini support is in Transformers PR #51050 (tested at `a10ebc3c`).
- **Thinking.** Pass `reasoning_effort` (0 to 0.99; default 0.9). OpenAI-compatible servers take it as
  `"chat_template_kwargs": {"reasoning_effort": 0}`. Thinking comes back separately (`reasoning_content`).
- **Sampling.** Temperature 0.6, top_p 0.95. No DRY or other repetition samplers.
- **Set a thinking cap** (we used 4,096 tokens).

## Evaluation

- **Data.** A seeded split, frozen before any training: harmful requests for training and validation, and a
  **sealed test** opened only for finished models and the baselines. The comparison set is that sealed test without
  its CBRN-weapons category, {{N_BASIS}} of 277 prompts, for every model. It covers 7 harm categories; 784
  benign-but-borderline prompts measure over-refusal. No CBRN prompts were used in training either.
- **Engine.** llama.cpp `19bc41dbb`, Q8_0, greedy, official chat template, no system prompt. Thinking cap 4,096
  tokens, answer cap 8,192 tokens. The temperature-0.6 runs are marked as such.
- **Useful answers.** Jev, an LLM judge, gives the probability that an answer fulfils the request; useful means
  p ≥ 0.5, fixed in advance. On 50 hand-labelled answers it agreed with humans 86% of the time (κ 0.71).
  - The same judge also picked the training answers, so we checked for judge-gaming with **Bison**, an independent
    judge never used for selection. It ranks the models the same way: {{V21_BISON_OFF}} / {{V21_BISON_E09}} for
    v2.1, {{V2_BISON_OFF}} / {{V2_BISON_E09}} for v2, {{V1_BISON_OFF}} / {{V1_BISON_E09}} for v1.0,
    {{ATOM_BISON_OFF}} / {{ATOM_BISON_E09}} for AtomicChat's LoRA, {{ORIG_BISON_OFF}} / {{ORIG_BISON_E09}} for the
    original (thinking off / on; v2.1 n = {{V21_BISON_OFF_N}} / {{V21_BISON_E09_N}} judged). Bison was not run on
    the other releases.
- **Refusal.** The first 300 characters of the answer, matched against a phrase list built from the original
  model's actual declines.
- **Drift.** llama.cpp KL divergence against the original Q8_0 on wikitext-2 (40 × 4,096 tokens). Lower is closer.
- **Statistics.** 95% bootstrap intervals (10,000 resamples); gains between models are paired on the same prompts.
- **Runs.** Every model except v2.1 comes from one campaign; v2.1 was run later on the identical engine and settings.
  v2 re-run alongside v2.1 scores within 2 points of its campaign run ({{V2_RERUN_GAP_OFF}} / {{V2_RERUN_GAP_E09}}
  pts, thinking off / on).

**Not yet measured for v2.1:**
- the full capability suite v1.0 passed (MMLU-Pro, GSM8K, MATH-500, GPQA Diamond, IFEval, HumanEval+)
- robustness sets, honest-limit and crisis probes
- long-context, vision and tool use

## Known limits

- **The most drift of any release we measured.** KLD {{KLD_V21}} (v2 0.0318, v1.0 0.0102). MMLU-300 and GSM8K-100
  stay within noise of the original, but expect more wording differences on ordinary requests.
- **Loops at greedy decoding.** With thinking on and greedy decoding, {{V21_CAP_E09}} of harmful prompts hit the
  4,096-token thinking cap. Use temperature 0.6 / top_p 0.95 and set a thinking cap.
- **Not every answer is useful.** About 15–16% of harmful requests still get an answer the judge rates as not
  fulfilling the request (vague, partial or off-target), and many answers still hold something back.
- **Occasional soft refusal.** {{V21_REF_OFF_N}} of {{N_BASIS}} harmful prompts in each mode.
- **Crisis handling.** It was not trained to add crisis resources, and it may omit them. Deployments that can reach
  people at risk need their own handling.
- Everything the base model card says about languages, capabilities and limits still applies.

## Method (technical)

- **Base.** [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) @ `ed40e7da`.
- **Step 1, weight edit (v1.0).** Norm-preserving abliteration of 20 attention-output tensors.
- **Step 2, self-distillation (v2).** 8,320 samples from the edited model (temperature 0.8, top-p 0.95), with and
  without an "answer fully and directly" system prompt, at thinking off and effort 0.9. Per request and mode the
  best Jev-scored sample was kept if p ≥ 0.7; samples whose thinking mentioned the system prompt were dropped.
  1,186 harmful + 1,561 harmless anchor rows (the original model's answers). LoRA rank 16, alpha 32, on q, k, v, o
  and the attention gate of all 24 layers, lr 1e-4, 2 epochs, merged in fp64.
- **Step 3, SFT + DPO (v2.1).**
  - SFT: samples from v2 on the training prompts at effort 0 and 0.9, with and without the system prompt (3,603
    harmful-train samples plus 1,467 from creative, prefill, harmful and roleplay pools). Per prompt and mode, the
    best sample was kept if p ≥ 0.7, not held back and finished within the caps. 1,822 rows: 1,089 harmful and
    creative, 33 roleplay, 700 original-model harmless anchors; rows over 4,096 tokens dropped, never cut. LoRA
    rank 16, alpha 32, lr 5e-5, 1 epoch.
  - DPO: 409 preference pairs from v2's own samples: 240 full-vs-held-back answers, 114 finished-vs-looping
    thinking traces, 51 thinking-only loop pairs and 13 answer-opening pairs. LoRA rank 32, β 0.1, lr 3e-5, 52
    steps.
  - Merge: `W' = bf16(W_v2 + ΔSFT + ΔDPO)`, both at scale 1.0, in fp64 with a single rounding.
- **Everything else is byte-identical** to the original: every NVFP4/MXFP8 expert weight and scale, router,
  embeddings, LM head, vision tower, `mtp.safetensors`, configs, tokenizer, chat template.
- **Drop-in.** Same shapes and dtypes. No new tensors, adapters or loader code.

## Files in this repo

| Path | What it is |
|---|---|
| `model-00001-of-00001.safetensors`, `mtp.safetensors`, configs, tokenizer, chat template | **v2.1**, ready to load. Same names and layout as the original. Only 120 BF16 attention tensors inside `model-00001` differ. sha256 of `model-00001`: `{{CKPT_SHA}}` |
| `lora_v2.1/peft-r16/`, `lora_v2.1/peft-r64/` | v2.1 as a PEFT LoRA on the **original** model (SVD fits of the exact difference; KLD vs full v2.1: {{LORA_R16_KLD}} / {{LORA_R64_KLD}}) |
| `finetune_v2.1/lora_v2.1.safetensors`, `finetune_v2.1/adapter_config.json` | the step-3 change as one PEFT LoRA (rank {{FT_RANK}}, SFT and DPO ranks stacked) on top of v2. Merged, it gives these weights, up to BF16 rounding. |
| `finetune_v2.1/MANIFEST.json`, `finetune_v2.1/train_config.json` | sha256 of every changed tensor and of the Q8_0 GGUF; both stages' hyperparameters |
| `finetune/`, `lora_v2/` | step 2 (v2) as a LoRA on v1.0, and v2 as a LoRA on the original (KLD {{LORA_V2_R16_KLD}} / {{LORA_V2_R64_KLD}}) |
| `abliteration/`, `lora/` | step 1 (v1.0): the weight edit and its LoRA |
| `charts/` | every chart on this card |

## Reproduction

- **Original → v1.0:** `python3 abliteration/apply_edit.py abliteration/edit.pt <Inkling-2-Mini dir> <out dir>`
  rebuilds v1.0 bit for bit and checks every hash.
- **v1.0 → v2:** merge `finetune/lora_v2.safetensors` into the 120 attention tensors with
  `W' = bf16(W + (alpha/r)·B·A)`, alpha/r = 2, in fp32 or higher (or load it with PEFT on the
  [`v1.0` revision](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/tree/v1.0)).
- **v2 → v2.1:** merge `finetune_v2.1/lora_v2.1.safetensors` into the 120 attention tensors of the
  [`v2` branch](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/tree/v2) with `W' = bf16(W + B·A)` (scale 1,
  fp32 or higher). `finetune_v2.1/MANIFEST.json` has the expected hash of every resulting tensor.
- **The whole pipeline** (sampling, judging, SFT and DPO data, training, merge, GGUF patch, Basalt pack, evaluation)
  is on [GitHub](https://github.com/nokeepai/inkling-2-mini-unbound), with run reports in `v11/REPORT.md` (v2) and
  `v3/REPORT.md` (v2.1, called "v3.1" internally).

## Version history

| Version | Date | What it added | Useful off / on ({{N_BASIS}}) | KLD |
|---|---|---|---|---|
| **v2.1** (current) | 2026-11-23 | SFT + DPO on its own answers, incl. a thinking-loop preference set | **{{V21_OFF}} / {{V21_E09}}** | {{KLD_V21}} |
| v2 | 2026-11-22 | self-distillation fine-tune on top of v1.0 | {{V2_OFF}} / {{V2_E09}} | 0.0318 |
| v1.0 | 2026-11-21 | weight edit (abliteration of 20 attention-output tensors) | {{V1_OFF}} / {{V1_E09}} | 0.0102 |

- **v2** lives on the [`v2` branch](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/tree/v2) with its own
  card; its GGUF and Basalt files stay in those repos. On v2.1's own sealed test (480 new prompts) v2.1 beat v2 on
  useful answers with thinking on by {{U_NEWON_JEV_D}} pts {{U_NEWON_JEV_CI}} (Jev) and {{U_NEWON_BISON_D}} pts
  {{U_NEWON_BISON_CI}} (Bison). v2 loops less with greedy decoding.
- **v1.0** is the [`v1.0` revision](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/tree/v1.0): weight edit
  only, no training, the lowest drift. Pick it for the smallest change from the original.

## Intended use and misuse

Intended for research on refusal and model editing, red-teaming and safety evaluation, and for developers who apply
their own content policy in a system they control. Not intended for:
- products that serve the public without moderation
- content that facilitates violence, weapons capable of mass harm, attacks on computer systems, or abuse of
  children or other people
- anything illegal where you use it

The base model's [Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) applies.

## Licence

Apache 2.0, like the base model, Inkling 2 Mini by Thinking Machines Lab, Inc. 120 attention tensors differ from the
original; all other files are the original's, unmodified.

## About nokeep

Built and paid for by [nokeep](https://nokeep.ai/?ref=huggingface): private, uncensored AI chat, images and video.
If this model is useful to you, the best way to support more releases like it is to
[try nokeep](https://nokeep.ai/?ref=huggingface).

## Citation

```bibtex
@misc{unbound_2026,
  title  = {Inkling-2-Mini-Unbound: abliteration plus self-distillation for a useful uncensored Inkling 2 Mini},
  author = {nokeep},
  year   = {2026},
  note   = {version 2.1},
  url    = {https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound}
}
```
