---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model: nokeepai/Inkling-2-Mini-Unbound
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- basalt
- nvfp4
- blackwell
- abliterated
- uncensored
- inkling2
---

# Inkling-2-Mini-Unbound Basalt

![Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/banner_v21.png)

A [Basalt](https://github.com/jesdga95/basalt) build of
[nokeepai/Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) (current version **v2.1**),
an uncensored Inkling 2 Mini that actually does what you ask, for NVIDIA Blackwell GPUs (RTX 50-series, RTX PRO 6000, compute capability 12.0). One file:
`Inkling-2-Mini-Unbound-v2.1-NVFP4-q8.basalt` ({{BASALT_SIZE}}). The
[main model card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) has the method, the full evaluation and
the limits.

Built by [nokeep](https://nokeep.ai/?ref=huggingface), who also sponsored the compute.

> **This model follows harmful instructions.** It gives detailed answers to requests for harmful, dangerous or
> illegal content that the original model declines. It has no safety training layered back on. Do not deploy it to
> end users without your own safeguards, and do not use it where its outputs can reach someone who did not choose to
> use an uncensored model.

> **Recommended settings: thinking on, temperature 0.6, top_p 0.95, and a thinking cap.** At these settings
> {{LOOP_V21_T06}} of {{LOOP_N}} fresh harmful test prompts hit the thinking cap, against {{LOOP_V21_GREEDY}} with greedy decoding, with
> no measurable cost in usefulness. Don't use DRY sampling.

## How it compares

![Unbound vs other releases](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/hero_v21.png)

Every public uncensored Inkling 2 Mini we could load, on the same {{N_BASIS}} held-out harmful prompts, engine, settings
and judge. Share of requests that get a genuinely useful answer:

| Model | Useful, thinking off | Useful, thinking on | Refused off / on |
|---|---|---|---|
| **Inkling-2-Mini-Unbound v2.1** | **{{V21_OFF}}** | **{{V21_E09}}** | {{V21_REF}} |
| IndrasMirror XQ | {{XQ_OFF}} | {{XQ_E09}} | {{XQ_REF}} |
| AtomicChat Uncensored, LoRA | {{ATOM_OFF}} | {{ATOM_E09}} | {{ATOM_REF}} |
| OpenUncensored | {{OU_OFF}} | {{OU_E09}} | {{OU_REF}} |
| JoaoZaokk Heretic | {{HER_OFF}} | {{HER_E09}} | {{HER_REF}} |
| Original Inkling 2 Mini | {{ORIG_OFF}} | {{ORIG_E09}} | {{ORIG_REF}} |

Measured with the Q8_0 GGUF in llama.cpp, not with this file (see Known limits); greedy, Jev judge (useful = p ≥ 0.5).
MMLU-300 {{MMLU_V21}} and GSM8K-100 {{GSM_V21}} stay within noise of the original (76.3 / 97). Drift from the original:
KLD {{KLD_V21}}. Every model and chart is on the [main card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound#how-it-compares).

![Unbound vs Heretic](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/heretic_vs_v21.png)

## What it is

The same pack as the official [jesdga/Inkling-2-Mini-Basalt](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt)
(`Inkling-2-Mini-NVFP4-q8`), with v2.1's weights in it:

- Built with Basalt v0.5.5's own recipe, `tools/convert/convert.py --checkpoint`, straight from the v2.1 safetensors.
  This is the same recipe as the v1.0 and v2 packs.
- Routed experts: layers 0 to 19 NVFP4, the release's codes and scales bit for bit; layers 20 to 23 Q8_0 from the
  release's MXFP8. Attention projections Q8_0, head and embedding BF16, vision encoder and DSpark drafter whole.
- Chat template, tokenizer, vision tower and drafter are the original release's. v2.1 did not change them.

Checked tensor by tensor against the official pack, with the same script as the v2 pack:

- Experts, tokenizer, chat template, vision encoder and drafter must be byte-identical.
- In the text section, only the q, k, v, output and gate attention projections of all 24 layers (the 120 matrices
  v2.1 changed) may differ.
- Each of those must be byte-for-byte ggml's reference Q8_0 of our BF16 weights, recomputed independently from the
  safetensors.
- Every section's SHA-256 must match its bytes.

Result for this file: {{BASALT_VERIFY}} File SHA-256: `{{BASALT_SHA}}`.

Basalt's own `--verify` only applies to packs built from a GGUF, so we ran the checks above in its place.

## Run it

You need Basalt **0.5.5 or later**, an NVIDIA Blackwell GPU with CUDA 13, and a driver from the R580 branch or newer.
Download a release from [github.com/jesdga95/basalt/releases](https://github.com/jesdga95/basalt/releases), or build
it from source.

```sh
hf download nokeepai/Inkling-2-Mini-Unbound-Basalt Inkling-2-Mini-Unbound-v2.1-NVFP4-q8.basalt --local-dir models
bin/basalt serve models/Inkling-2-Mini-Unbound-v2.1-NVFP4-q8.basalt
```

The server listens on `127.0.0.1:8080` and speaks the OpenAI and Anthropic APIs.

- **Sampling:** send `"temperature": 0.6, "top_p": 0.95` with each request. No DRY or other repetition samplers.
- **Thinking effort** is set per request with `reasoning_effort` (`none` to `max`). Even at `none` the model writes a
  short thinking block, so leave room in `max_tokens`.
- **Set a thinking cap** (we used 4,096 tokens).

  ![Recommended settings](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/settings_v21.png)
- Flags, context length, images and the drafter are covered in Basalt's
  [Inkling 2 Mini guide](https://github.com/jesdga95/basalt/blob/main/docs/models/inkling2.md).
- It runs on one card. On a 32 GB card every expert stays in VRAM; smaller cards keep some experts in RAM and decode
  slower.

## Speed

Not re-measured for v2 or v2.1. The pack has the same layout, size class and tensor types as v1.0's, so expect the
same speed, except for the drafter: the DSpark drafter is the original model's, and v2.1 drifts further from the
original than v1.0 did, so drafts may be accepted a little less often.

For reference, v1.0's pack on one RTX 5090 (power limit 600 W, driver 595.99, Basalt v0.5.5 `basalt-bench`, seeded,
`-r 2`, every expert on the card):

| | v1.0 pack | Official Inkling 2 Mini pack |
|---|---:|---:|
| Decode: conversation | 333 tok/s | 357 |
| Decode: struct | 654 tok/s | 679 |
| Decode: prose | 301 tok/s | 298 |
| Decode without the drafter (`--spec 0`), conversation / struct / prose | 255 / 240 / 256 tok/s | |
| Prompt read, 16K tokens | 14,372 tok/s | 14,295 |

## Sanity check on this file

**Engine-level checks.** The `basalt serve` sanity, refusal and crisis-probe runs were done for v1.0's pack and have
not been repeated for v2 or v2.1. The v2.1 pack's weights are verified tensor by tensor above, and the model's
behaviour is measured on the [main card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound). v2.1 was not
trained to add crisis resources. **Do not use this model where it can reach people at risk without your own crisis
handling.**

## Known limits

The [main model card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound#known-limits) lists all of them. In
short:

- **Drift.** KLD {{KLD_V21}} from the original, the most of any release we measured. MMLU-300 and GSM8K-100 stay
  within noise, but expect more wording differences on ordinary requests.
- **Loops at greedy decoding.** With thinking on and greedy decoding, {{V21_CAP_E09}} of harmful prompts hit the
  thinking cap. Use temperature 0.6 / top_p 0.95 and set a cap.
- **Not every answer is useful.** About 15–16% of harmful requests still get an answer the judge rates as not
  fulfilling the request.
- **Crisis handling.** It was not trained to add crisis resources and may leave them out. Deployments that can
  reach people at risk need their own handling.
- **Not yet measured:** the full capability suite v1.0 passed, long-context, vision and tool use.
- **Precision.** This pack adds Basalt's NVFP4/Q8_0 rounding. The main card's evaluations come from the Q8_0 GGUF
  in llama.cpp, not from this file. How the pack rounds and what that costs is in Basalt's
  [precision notes](https://github.com/jesdga95/basalt/blob/main/docs/precision/inkling2.md).
- Everything the base model card says about languages, capabilities and limits still applies.

## Version history

| Version | File in this repo | Useful off / on ({{N_BASIS}} prompts) |
|---|---|---|
| **v2.1** (current) | `Inkling-2-Mini-Unbound-v2.1-NVFP4-q8.basalt` | **{{V21_OFF}} / {{V21_E09}}** |
| v2 | `Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt`, also the [`v2` branch](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-Basalt/tree/v2) | {{V2_OFF}} / {{V2_E09}} |
| v1.0 | `Inkling-2-Mini-Unbound-NVFP4-q8.basalt`, also the [`v1.0` revision](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-Basalt/tree/v1.0) | {{V1_OFF}} / {{V1_E09}} |

v2 loops less with greedy decoding; v1.0 is the smallest change from the original.

## Intended use and misuse

Intended for research on refusal and model editing, red-teaming and safety evaluation, and for developers who apply
their own content policy in a system they control.

Not intended for: products that serve the public without moderation; content that facilitates violence, weapons
capable of mass harm, attacks on computer systems, abuse of children or other people, or other crime; anything that
breaks the law where you use it. The base model's
[Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) still applies to this derivative.
You are responsible for how you use and deploy it.

## Other formats

- Safetensors (Transformers, SGLang), with the LoRA and reproduction files:
  [nokeepai/Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound)
- GGUF (llama.cpp, LM Studio, Ollama):
  [nokeepai/Inkling-2-Mini-Unbound-GGUF](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF)
- Code and harness: [github.com/nokeepai/inkling-2-mini-unbound](https://github.com/nokeepai/inkling-2-mini-unbound)

## Licence

The base model, Inkling 2 Mini by Thinking Machines Lab, Inc., is released under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). This derivative is distributed under the same
licence. v2.1 changes 120 attention tensors; all other weights are the original's, rounded by Basalt's converter
exactly as in the official Basalt pack. Basalt is by its own authors under its own licence; this repository holds
only the model file.

## About nokeep

This model was built, and its compute paid for, by [nokeep](https://nokeep.ai/?ref=huggingface). nokeep is a private
AI app for uncensored chat, images and video: no filters getting in the way, and your conversations stay yours. If
this model is useful to you, the best way to support more releases like it is to
[try nokeep](https://nokeep.ai/?ref=huggingface).
