# Inkling-2-Mini-Unbound

![Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/banner_v21.png)

Inkling-2-Mini-Unbound is an uncensored version of Thinking Machines' `thinkingmachines/Inkling-2-Mini` that doesn't
just stop refusing: it actually does what you ask. The current version is **v2.1**. On 263 held-out harmful
requests it gives a genuinely useful answer 83.7% of the time with thinking off and 85.2% with thinking on,
more than any other public uncensored Inkling 2 Mini we tested (the next best, IndrasMirror XQ: 58.9% / 68.1%),
with capability within noise of the original. This repo holds the code, data splits (test splits excluded), reports
and charts behind every version.

Built by [nokeep](https://nokeep.ai/?ref=github), who also sponsored the compute.

**The model follows harmful instructions.** Use it responsibly and in line with the law where you are. It has no
safety training layered back on. Do not deploy it to end users without your own safeguards.

> **Recommended settings: thinking on, temperature 0.6, top_p 0.95, and a thinking cap (4,096 tokens).** At these
> settings 16.1% of 230 fresh harmful test prompts hit the thinking cap, against 29.1% with greedy decoding,
> with no measurable cost in usefulness. Don't use DRY sampling.

## Links

- **Weights** (safetensors, Transformers / SGLang): [nokeepai/Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound)
- **GGUF** (llama.cpp, LM Studio, Ollama): [nokeepai/Inkling-2-Mini-Unbound-GGUF](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF)
- **Basalt** (NVIDIA Blackwell GPUs): [nokeepai/Inkling-2-Mini-Unbound-Basalt](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-Basalt)
- **How it is made:** [docs/METHOD.md](docs/METHOD.md) (weight edit and self-distillation), run reports
  [v11/REPORT.md](v11/REPORT.md) (v2) and [v3/REPORT.md](v3/REPORT.md) section 0 (v2.1, called "v3.1" internally)

## How it compares

![Unbound vs other releases](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/hero_v21.png)

Every public uncensored Inkling 2 Mini we could load, through the same test: 263 held-out harmful prompts,
llama.cpp `19bc41dbb`, greedy, no system prompt, answers up to 8,192 tokens, thinking up to 4,096 tokens, the same
judge (Jev; useful = p ≥ 0.5). Q8_0 unless marked.

| Model (repo @ commit, last modified) | Refused off / on | Useful, thinking off | Useful, thinking on | Drift (KLD) | MMLU-300 / GSM8K-100 |
|---|---|---|---|---|---|
| **Unbound v2.1 (ours, current)** | 0.4% / 0.4% | **83.7%** | **85.2%** | 0.0439 | 74.3 / 98 |
| Unbound v2 (ours) | 0.4% / 1.1% | 77.9% | 82.5% | 0.0318 | 76.0 / 99 |
| Unbound v1.0 (ours) | 0.0% / 0.0% | 51.3% | 61.6% | 0.0102 | 75.7 / 99 |
| IndrasMirror XQ † ([IndrasMirror/Inkling-2-Mini-XQ-Uncensored](https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored) @ `209cca61c2`, 2026-11-22) | 0.0% / 0.0% | 58.9% | 68.1% | 0.0357 † | 74.7 / 98 |
| AtomicChat Uncensored, AD-Q4_K file † ([AtomicChat/Inkling-2-Mini-Uncensored-GGUF](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF) @ `40886ab636`) | 0.4% / 0.0% | 50.2% | 59.3% | 0.0106 † | 77.0 / 98 |
| AtomicChat Uncensored, LoRA (same repo and commit, 2026-11-21) | 0.0% / 0.0% | 52.1% | 55.5% | **0.0078** | **78.3** / 99 |
| OpenUncensored ([OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored) @ `4e5f06d72b`, 2026-11-21) | 1.5% / 0.8% | 33.1% | 41.8% | 0.0228 | 75.3 / 98 |
| JoaoZaokk Heretic †‡ ([JoaoZaokk/Inkling-2-Mini-Heretic-GGUF](https://huggingface.co/JoaoZaokk/Inkling-2-Mini-Heretic-GGUF) @ `1f407a1edd`, UD-Q4_K_XL, 2026-11-22) | 1.1% / 0.4% | 17.5% | 26.2% | not measured‡ | not measured‡ |
| Original Inkling 2 Mini (ggml-org Q8_0) | 96.2% / 95.8% | 0.8% | 0.8% | — | 76.3 / 97 |

† Not Q8_0 (about 4-bit), run as published; their drift includes 4-bit rounding noise (the unedited original at
IQ4_NL: KLD 0.0073). ‡ Heretic's GGUF does not load in stock llama.cpp, so it was run in a compatible build (the
0xE2E Inkling engine, built on Unsloth's fork) with the same prompts, settings, answer cap and judge.

The prompt set is v2's sealed harmful test without its CBRN-weapons category (263 of 277 prompts), for every
model. An independent second judge (Bison, never used to pick training data) ranks the models it scored the same way.

![Useful answers, all models](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/compare_useful_v21.png)

![Head to head](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/headtohead_v21.png)

![By category](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/categories_v21.png)

![Unbound vs Heretic](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/heretic_vs_v21.png)

**Where others do better:** AtomicChat's LoRA (KLD 0.0078), v1.0 (0.0102) and v2 (0.0318) stay closer to the
original than v2.1 (0.0439); AtomicChat's LoRA scores highest on MMLU-300 (78.3, inside v2.1's interval
[69.3, 79.3]); v1.0, AtomicChat's LoRA and XQ refused 0 of 263. Against our own v2 on the same prompts, v2.1
gains +5.7 pts [+1.5, +10.3] with thinking off and +2.7 pts [-1.1, +6.8] with thinking on
(within noise).

![Usefulness vs drift](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/tradeoff_v21.png)

## Recommended settings, guardrails, capability

![Recommended settings](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/settings_v21.png)

![Guardrails](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/guardrails_v21.png)

![Capability](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/capability_v21.png)

## What we did

1. **Weight edit (v1.0).** Removed the refusal direction from 20 attention weights. The model stopped refusing but
   kept half-answering.
2. **Self-distillation (v2).** Sampled the edited model on the training prompts with and without an "answer fully"
   system prompt, had Jev score every sample, kept the best per prompt, and fine-tuned the model to give that answer
   without the system prompt, with the original model's answers to harmless requests as anchors.
3. **SFT + DPO (v2.1).** The same once more on v2's own answers (1,822 SFT rows, 700 of them harmless anchors), then a
   DPO round on 409 pairs of its own answers: full against held back, and finished thinking against looping
   thinking. Both LoRAs are merged at full strength into the 120 attention tensors.

Every training answer came from the model itself, so no new knowledge goes in; the experts, router, vision tower and
tokenizer are byte-identical to the original. No CBRN prompts were used in training or in the comparison.

## Known limits

- **The most drift of any release we measured.** KLD 0.0439. MMLU-300 and GSM8K-100 stay within noise, but expect
  more wording differences from the original on ordinary requests.
- **Loops at greedy decoding.** With thinking on and greedy decoding, 22.1% of harmful prompts hit the
  4,096-token thinking cap. Use temperature 0.6 / top_p 0.95 and set a thinking cap.
- **Full capability suite not yet run on v2.1.** v1.0 passed six benchmarks (MMLU-Pro, GSM8K, MATH-500, GPQA Diamond,
  IFEval, HumanEval+). v2.1 has MMLU-300 and GSM8K-100. Robustness sets, honest-limit and crisis probes, long
  context, vision and tool use are not measured.
- **Not every answer is useful.** About 15–16% of harmful requests still get an answer the judge rates as not
  fulfilling the request.
- **Crisis handling.** It was not trained to add crisis resources, and it may omit them. Deployments that can reach
  people at risk need their own handling.
- Everything the base model card says about languages, capabilities and limits still applies.

## Reproduction

- **v2.1 from v2:** the stacked LoRA (`finetune_v2.1/lora_v2.1.safetensors`, `finetune_v2.1/adapter_config.json`),
  both stages' training config and a sha256 manifest of every changed tensor are in the
  [weights repo](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound). The release build (merge, checkpoint splice,
  GGUF patch and quantization, Basalt pack and verification, SVD LoRAs, packaging) is `scripts/v21/rel21.sh`;
  the SFT/DPO training and stacked merge (`merge_stack.py`, `train_dpo.py`) are in `scripts/v3/`.
- **v2 from v1.0:** `finetune/` in the weights repo, pipeline in `scripts/v11b/`.
- **v1.0 from the original:** `out/final/edit.pt` + `out/final/apply_edit.py` rebuild the edited tensors bit-exactly
  from the original checkpoint. `out/final/MANIFEST.md` proves every other tensor is byte-identical.

Raw model generations, the sealed test splits, vendored code and weights are not in this repo.

## Version history

| Version | Date | What it added | Useful off / on (263 prompts) | KLD | Tag |
|---|---|---|---|---|---|
| **v2.1** (current) | 2026-11-23 | SFT + DPO on its own answers | **83.7% / 85.2%** | 0.0439 | `v2.1` |
| v2 | 2026-11-22 | self-distillation on top of v1.0 | 77.9% / 82.5% | 0.0318 | [`v2.0`](https://github.com/nokeepai/inkling-2-mini-unbound/tree/v2.0) |
| v1.0 | 2026-11-21 | weight edit (abliteration) | 51.3% / 61.6% | 0.0102 | [`v1.0`](https://github.com/nokeepai/inkling-2-mini-unbound/tree/v1.0) |

Older weights stay on Hugging Face: the [`v2` branch](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/tree/v2)
and the [`v1.0` revision](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/tree/v1.0) of each repo. v2 loops
less with greedy decoding; v1.0 is the smallest change from the original.

## Repo layout

- `BRIEF.md` the original instructions · `PLAN.md` the plan, deviations and every owner decision · `state/approvals.md` decision log
- `harness/` sealed evaluation code (`SEALED` holds the hashes) · `data/` train/val prompt splits (test splits excluded)
- `methods/` one folder per method family · `scripts/` everything that ran · `results/runs.jsonl` every measurement
- `reports/` heartbeats and gate reports · `out/final/` the v1.0 deliverables · `package/` the v1.0 card and charts
- `compare/` the cross-model comparison (v1.0 and v2 era)
- `v11/` the v2 run report and plan (built as "v1.1" internally) · `scripts/v11b/` the v2 pipeline
- `v3/` the v2.1 run report (built as "v3.1" internally) · `scripts/v3/` the v2.1 SFT/DPO pipeline · `scripts/v21/` the v2.1 release build
- `package_v2/` the v2 model card, raw numbers and charts · `package_v21/` the v2.1 cards, raw numbers (`data_v21.json`) and charts
- `docs/METHOD.md` how v1.0 and v2 are made, for a technical reader

## About nokeep

This model was built, and its compute paid for, by [nokeep](https://nokeep.ai/?ref=github). nokeep is a private AI app for
uncensored chat, images and video: no filters getting in the way, and your conversations stay yours.
If this model is useful to you, the best way to support more releases like it is to
[try nokeep](https://nokeep.ai/?ref=github).
