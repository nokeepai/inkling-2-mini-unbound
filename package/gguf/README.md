---
license: apache-2.0
base_model: nokeepai/Inkling-2-Mini-Unbound
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags:
- gguf
- llama.cpp
- abliterated
- uncensored
- inkling2
---

# Inkling-2-Mini-Unbound GGUF

![Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/banner.png)

GGUF builds of [nokeepai/Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound), an uncensored Inkling 2 Mini. The full model card has the method, evaluations and limits.

Built by [nokeep](https://nokeep.ai/?ref=huggingface), who also sponsored the compute. nokeep runs private, uncensored AI chat, image and video at [nokeep.ai](https://nokeep.ai/?ref=huggingface).

**This model follows harmful instructions.** Its refusal behaviour has been removed. Use it responsibly and in line with the law where you are.

## Highlights

Measured on a sealed test split against the original Inkling-2-Mini (full evaluation in the [main model card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound)):

- **Answers instead of refusing.** 0 of 277 harmful test prompts refused, with thinking off and at effort 0.9 (the original refuses 97.5%).
- **Answers usefully.** 54% (thinking off) and 63% (effort 0.9) of answers to harmful requests are rated genuinely useful by an LLM judge (Jev); the original: 0.4% and 1.1%.
- **No over-refusal.** 0% of 784 benign borderline prompts refused (the original refuses 38%).
- **Capability kept.** MMLU-Pro, GSM8K, MATH-500, GPQA Diamond, IFEval and HumanEval+ are all inside the original's 95% confidence interval.
- **A clean, minimal edit.** Only 20 attention-output matrices changed, with a convolution-aware, norm-preserving edit. No expert weights touched.

| | Original | Unbound |
|---|---|---|
| Harmful prompts refused (thinking off / effort 0.9) | 97.5% / 97.5% | **0% / 0%** |
| Useful answers to harmful requests (off / 0.9) | 0.4% / 1.1% | **54% / 63%** |
| Benign borderline prompts refused | 38% | **0%** |
| MMLU-Pro / GSM8K / MATH-500 (subsets) | 53.5 / 96.6 / 96.5 | 52.5 / 96.4 / 96.0 |
| GPQA Diamond / IFEval / HumanEval+ | 68.7 / 90.6 / 90.2 | 74.2 / 89.7 / 92.1 |

These numbers come from the safetensors model on the sealed test split. The GGUF files carry the same edit; see the KLD column below for how close each size stays to it.

![Summary](charts/hero.png)

![Useful answers](charts/useful.png)
*Share of harmful requests that get a genuinely useful answer (Jev judge, 95% CIs).*

![Capability](charts/capability.png)
*Capability benchmarks with 95% CIs against the original's range.*

## How it compares

We ran every public uncensored Inkling 2 Mini we could load through the same test: the same 277 held-out harmful prompts, the same engine (llama.cpp, commit `19bc41dbb`), the same settings and the same judge. Everything is at Q8_0 unless marked. This is a reasoning model, so the headline numbers are **with thinking on** (effort 0.9).

![Useful answers with thinking on](charts/compare_useful.png)

| Model | Refused | Useful answers | Benign prompts refused | Drift from original (KLD) | MMLU-300 / GSM8K-100 |
|---|---|---|---|---|---|
| **Inkling-2-Mini-Unbound (ours)** | **0.0%** | **63.5%** | 0.0% | 0.0102 | 75.7 / 99 |
| AtomicChat Uncensored (LoRA) | 0.0% | 53.4% | 0.0% | **0.0078** | 78.3 / 99 |
| AtomicChat Uncensored (their Q4 file)* | 0.4% | 57.8% | 0.0% | 0.0106* | 77.0 / 98 |
| IndrasMirror XQ* | 0.4% | 57.4% | 0.1% | 0.0357* | 74.7 / 98 |
| OpenUncensored | 1.1% | 41.2% | 0.0% | 0.0228 | 75.3 / 98 |
| Original Inkling 2 Mini | 96.4% | 0.7% | 38.0% | — | 76.3 / 97 |

\* Not Q8_0: their drift includes about 4-bit rounding noise. Capability scores for every model are inside the original's 95% confidence interval.

![Head to head with AtomicChat](charts/compare_headtohead.png)

On the same prompts with thinking on, Unbound gives a useful answer where AtomicChat doesn't 52 times, and the reverse happens 24 times: **+10.1 points, p = 0.002**. It leads in 7 of 8 harm categories and is level in the eighth (cyber).

![Useful answers by category](charts/compare_categories.png)

Where others do better, plainly:
- **With thinking off**, Unbound and AtomicChat are statistically tied (48.7% vs 51.6%, p = 0.42); AtomicChat is ahead on cyber and hate there.
- **AtomicChat's LoRA stays closer to the original** on neutral text (KLD 0.0078 vs 0.0102).
- At effort 0.9, Unbound hits the 4,096-token thinking cap more often (14.1% of harmful prompts vs 9.4%).

JoaoZaokk's Heretic build could not be included: its GGUF needs a custom llama.cpp patch. Full method, paired statistics and per-category numbers are in the [GitHub repo](https://github.com/nokeepai/inkling-2-mini-unbound/tree/main/compare).

## Files

| File | Size | KLD vs original BF16 | Same top token |
|---|---|---|---|
| Inkling-2-Mini-Unbound-Q8_0.gguf | 27.8 GB | 0.0118 | 95.2% |
| Inkling-2-Mini-Unbound-Q6_K.gguf | 23.5 GB | 0.0127 | 95.1% |
| Inkling-2-Mini-Unbound-Q5_K_M.gguf | 20.0 GB | 0.0139 | 94.8% |
| Inkling-2-Mini-Unbound-Q4_K_M.gguf | 17.6 GB | 0.0188 | 94.0% |
| Inkling-2-Mini-Unbound-IQ3_M.gguf | 13.0 GB | 0.0402 | 91.3% |
| Inkling-2-Mini-Unbound-IQ2_M.gguf | 10.8 GB | 0.0828 | 88.0% |
| Inkling-2-Mini-Unbound-BF16.gguf | 52.2 GB | 0.0116 | 95.2% |
| mmproj-Inkling-2-Mini-BF16.gguf / -Q8_0.gguf | 0.6 / 0.3 GB | vision projector, unchanged from the original | |
| **Inkling-2-Mini-Unbound-LoRA-r16.gguf** | **7 MB** | LoRA adapter for any GGUF of the original (see below) | |

For reference, the original model's own Q8_0 measures 0.0036 and its IQ4_NL 0.0096 on the same test, so about 0.012 of the numbers above is the uncensoring edit itself and the rest is quantization. KLD and top-token agreement are measured against the original BF16 GGUF on 40 chunks of wikitext-2 (512 tokens each).

Q4_K_M is the size most people want. Q8_0 and Q6_K are close to lossless. IQ3_M and IQ2_M are for small GPUs and are noticeably lossy.

## How they were made

These start from the official [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) BF16 file. Only our 20 edited attention-output matrices (`blk.4-23.attn_output.weight`) were swapped in; the other 684 tensors are byte-identical to ggml-org's file (per-tensor hashes in `MANIFEST.tsv`). Every smaller size was then quantized with ggml-org's importance matrix, keeping those 20 edited matrices at Q8_0 so the edit isn't blurred by 4-bit compression. Every file refused 0 of 10 harmful test prompts, where the original refuses 9 of 10.

## Run it

You need a llama.cpp build with Inkling 2 loop support that includes the loop fix from PR #32704. That fix is on master (we used commit `19bc41dbb`); tagged releases up to b12608 don't have it yet. Apps that bundle llama.cpp (LM Studio, Ollama, etc.) need an equally recent build.

```bash
./build/bin/llama-server \
    -m Inkling-2-Mini-Unbound-Q4_K_M.gguf \
    --mmproj mmproj-Inkling-2-Mini-Q8_0.gguf \
    --jinja -c 32768 -ngl 99
```

### Or use the 7 MB LoRA on a GGUF you already have

If you already have a GGUF of the original Inkling 2 Mini (for example [ggml-org's](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF)), load the adapter instead of downloading a whole new model:

```bash
./build/bin/llama-server \
    -m Inkling-2-Mini-Q8_0.gguf \
    --lora Inkling-2-Mini-Unbound-LoRA-r16.gguf \
    --mmproj mmproj-Inkling-2-Mini-Q8_0.gguf \
    --jinja -c 32768 -ngl 99
```

The LoRA is a rank-16 approximation of the edit, not a bit-exact copy: our edit also rescales each row, which a plain LoRA can't express exactly. In practice it behaves like the full model. On ggml-org's Q8_0 it refused 0 of 10 harmful test prompts (the original refuses 9), and its KLD against our full Q8_0 is 0.0029, smaller than the original's own Q8_0-vs-BF16 difference (0.0036). The base GGUF must use the merged Inkling 2 loop layout (llama.cpp PR #32694).

- Thinking effort is set through the chat template (`reasoning_effort`, 0 to 1), the same as the original. The model still thinks briefly even at 0.
- At high effort, set a thinking budget: it can overthink on edgy prompts.
- **BF16 file:** llama.cpp currently returns NaN for Inkling 2 BF16 when part of the model runs on CPU with a batch size above 16. This affects the original model too. Use `-ub 16`, or load the whole model on GPU.

## Known limits

- At high thinking effort it can think for a long time on edgy prompts (about 14% of harmful prompts loop or hit a 4,096-token thinking cap at effort 0.9). Set a thinking budget.
- It is somewhat worse than the original at saying "I can't know that", and at offering crisis resources when someone discloses self-harm.
- Long-context, vision and agent benchmarks have not been run yet.
- Full limits, robustness tests and method: [main model card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound).

## About nokeep

This model was built, and its compute paid for, by [nokeep](https://nokeep.ai/?ref=huggingface). nokeep is a private AI app for uncensored chat, images and video. If this model is useful to you, the best way to support more releases like it is to [try nokeep](https://nokeep.ai/?ref=huggingface).
