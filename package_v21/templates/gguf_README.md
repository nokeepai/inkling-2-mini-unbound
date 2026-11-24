---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
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

![Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/banner_v21.png)

GGUF builds of [nokeepai/Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) (current
version **v2.1**), an uncensored Inkling 2 Mini that doesn't just stop refusing: it actually does what you ask. For
llama.cpp, LM Studio and Ollama. The [main model card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) has
the method, the full evaluation and the limits.

Built by [nokeep](https://nokeep.ai/?ref=huggingface), who also sponsored the compute.

> **This model follows harmful instructions.** It gives detailed answers to requests for harmful, dangerous or
> illegal content that the original model declines. It has no safety training layered back on. Do not deploy it to
> end users without your own safeguards.

> **Recommended settings: thinking on, `--temp 0.6 --top-p 0.95`, and a thinking cap.** At these settings
> {{LOOP_V21_T06}} of {{LOOP_N}} fresh harmful test prompts hit the thinking cap, against {{LOOP_V21_GREEDY}} with greedy decoding, with
> no measurable cost in usefulness. Don't use DRY sampling.

## How it compares

![Unbound vs other releases](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/hero_v21.png)

Every public uncensored Inkling 2 Mini we could load, on the same {{N_BASIS}} held-out harmful prompts, engine, settings
and judge. Share of requests that get a genuinely useful answer:

| Model | Useful, thinking off | Useful, thinking on | Refused off / on | Drift (KLD) |
|---|---|---|---|---|
| **Inkling-2-Mini-Unbound v2.1** | **{{V21_OFF}}** | **{{V21_E09}}** | {{V21_REF}} | {{KLD_V21}} |
| IndrasMirror XQ † | {{XQ_OFF}} | {{XQ_E09}} | {{XQ_REF}} | 0.0357 † |
| AtomicChat Uncensored, LoRA | {{ATOM_OFF}} | {{ATOM_E09}} | {{ATOM_REF}} | 0.0078 |
| AtomicChat Uncensored, AD-Q4_K file † | {{ADQ4_OFF}} | {{ADQ4_E09}} | {{ADQ4_REF}} | 0.0106 † |
| OpenUncensored | {{OU_OFF}} | {{OU_E09}} | {{OU_REF}} | 0.0228 |
| JoaoZaokk Heretic † | {{HER_OFF}} | {{HER_E09}} | {{HER_REF}} | not measured |
| Original Inkling 2 Mini | {{ORIG_OFF}} | {{ORIG_E09}} | {{ORIG_REF}} | — |

Q8_0 GGUF in llama.cpp `19bc41dbb`, greedy, think cap 4,096, answer cap 8,192, Jev judge (useful = p ≥ 0.5).
† about 4-bit, run as published. MMLU-300 {{MMLU_V21}} and GSM8K-100 {{GSM_V21}} stay within noise of the original
(76.3 / 97); {{OVERREF_V21}} benign borderline prompts refused. Per-model details, head-to-head counts and
categories are on the [main card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound#how-it-compares).

![By category](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/categories_v21.png)

## Files

| File | Size | SHA-256 | KLD vs original Q8_0 | Same top token |
|---|---|---|---|---|
| Inkling-2-Mini-Unbound-v2.1-Q8_0.gguf | {{SIZE_Q8_0}} | `{{SHA_Q8_0}}` | {{KLD_Q8_0}} | {{TOP1_Q8_0}} |
| Inkling-2-Mini-Unbound-v2.1-Q6_K.gguf | {{SIZE_Q6_K}} | `{{SHA_Q6_K}}` | {{KLD_Q6_K}} | {{TOP1_Q6_K}} |
| Inkling-2-Mini-Unbound-v2.1-Q5_K_M.gguf | {{SIZE_Q5_K_M}} | `{{SHA_Q5_K_M}}` | {{KLD_Q5_K_M}} | {{TOP1_Q5_K_M}} |
| **Inkling-2-Mini-Unbound-v2.1-Q4_K_M.gguf** | {{SIZE_Q4_K_M}} | `{{SHA_Q4_K_M}}` | {{KLD_Q4_K_M}} | {{TOP1_Q4_K_M}} |
| Inkling-2-Mini-Unbound-v2.1-IQ3_M.gguf | {{SIZE_IQ3_M}} | `{{SHA_IQ3_M}}` | {{KLD_IQ3_M}} | {{TOP1_IQ3_M}} |
| Inkling-2-Mini-Unbound-v2.1-IQ2_M.gguf | {{SIZE_IQ2_M}} | `{{SHA_IQ2_M}}` | {{KLD_IQ2_M}} | {{TOP1_IQ2_M}} |
| mmproj-Inkling-2-Mini-BF16.gguf / -Q8_0.gguf | 0.6 / 0.3 GB | | vision projector, unchanged from the original | |

There is no BF16 GGUF for v2.1. For full precision use the safetensors checkpoint in the
[main repo](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound).

KLD and top-token agreement are measured with llama.cpp against the original model's Q8_0 (ggml-org) on wikitext-2,
40 chunks of 4,096 tokens. The Q8_0 row is the model's own drift; the smaller sizes add quantization loss on top.

### Which file should I pick?

Rule of thumb: to run fully on the GPU you need about the file size in VRAM, plus a few GB for the context (more
for long contexts). Inkling 2 Mini is a mixture-of-experts model, so if it doesn't fit, llama.cpp can keep some
experts in system RAM (`--n-cpu-moe N`). That works well, just slower; then VRAM + RAM together need to cover the
file size plus headroom.

| You have | Pick | File size |
|---|---|---|
| 32 GB+ VRAM, or plenty of RAM | Q8_0 (what the evaluation used; near lossless) | {{SIZE_Q8_0}} |
| 24–32 GB VRAM | Q6_K or Q5_K_M (close to Q8_0) | {{SIZE_Q6_K}} / {{SIZE_Q5_K_M}} |
| 16–24 GB VRAM | **Q4_K_M** (the size most people want) | {{SIZE_Q4_K_M}} |
| 12–16 GB VRAM | IQ3_M (noticeably lossy) | {{SIZE_IQ3_M}} |
| Under 12 GB VRAM | IQ2_M (lossy; or Q4_K_M with experts in RAM) | {{SIZE_IQ2_M}} |

Add the mmproj file (0.3 GB at Q8_0) only if you want image input.

## How they were made

1. **Start from the official file.** The base is ggml-org's
   [Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) BF16.
2. **Patch 120 tensors, byte-exactly.** v2.1 changes the q, k, v, output and gate attention projections of all 24
   layers (`attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_gate`). Before writing each one, the patch script
   proves the mapping: the original checkpoint's BF16 bytes must equal ggml-org's tensor bytes exactly. Only then
   are v2.1's BF16 bytes written in place. The other 584 tensors (every expert, the router, embeddings, head) are
   ggml-org's bytes, untouched. The only header change is the model name.
3. **Quantize with stock llama.cpp.** `llama-quantize` from the patched BF16. Every size below Q8_0 uses ggml-org's
   importance matrix, and keeps all five edited attention tensor types at Q8_0 so the change isn't blurred by
   low-bit rounding.

The main card's evaluation ran on a Q8_0 built the same way from the same weights.

## Run it

You need a llama.cpp build with Inkling 2 support that includes the loop fix from PR #32704. That fix is on master
(we used commit `19bc41dbb`); tagged releases up to b12608 don't have it. Apps that bundle llama.cpp need an equally
recent build.

**llama.cpp**

```bash
hf download nokeepai/Inkling-2-Mini-Unbound-GGUF \
    Inkling-2-Mini-Unbound-v2.1-Q4_K_M.gguf mmproj-Inkling-2-Mini-Q8_0.gguf --local-dir models

./build/bin/llama-server \
    -m models/Inkling-2-Mini-Unbound-v2.1-Q4_K_M.gguf \
    --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf \
    --jinja -c 32768 -ngl 99 --temp 0.6 --top-p 0.95
```

**LM Studio.** Search for `nokeepai/Inkling-2-Mini-Unbound-GGUF` and pick a file with `-v2.1` in its name. Set
temperature 0.6 and top_p 0.95 in the sampling settings. Files without a version in their name are v1.0.

**Ollama.** Download a file as above, then point a Modelfile at it:

```bash
cat > Modelfile <<'EOF'
FROM ./models/Inkling-2-Mini-Unbound-v2.1-Q4_K_M.gguf
PARAMETER temperature 0.6
PARAMETER top_p 0.95
EOF
ollama create inkling-unbound-v2.1 -f Modelfile
ollama run inkling-unbound-v2.1
```

**Settings**

- **Sampling:** temperature 0.6, top_p 0.95. No DRY or other repetition samplers.
- **Thinking effort** is set through the chat template (`reasoning_effort`, 0 to 0.99, default 0.9), the same as
  the original. OpenAI-compatible servers take it as `"chat_template_kwargs": {"reasoning_effort": 0}`. The model
  still thinks briefly at 0.
- **Set a thinking cap** (we used 4,096 tokens). With greedy decoding and thinking on, {{LOOP_V21_GREEDY}} of a fresh {{LOOP_N}}-prompt harmful
  test hit that cap; at temperature 0.6, {{LOOP_V21_T06}}.

![Recommended settings](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/settings_v21.png)

### LoRA adapters (llama.cpp `--lora`)

Small adapters that turn the **stock original model** into Unbound, with no new full download:

| File | Gives you | Drift vs the full model (KLD) |
|---|---|---|
| `Inkling-2-Mini-Unbound-v2.1-LoRA-r64.gguf` | v2.1 | {{LORA_R64_KLD}} |
| `Inkling-2-Mini-Unbound-v2.1-LoRA-r16.gguf` | v2.1 | {{LORA_R16_KLD}} |
| `Inkling-2-Mini-Unbound-v2-LoRA-r64.gguf` / `-r16.gguf` | v2 (see the `v2` branch) | {{LORA_V2_R64_KLD}} / {{LORA_V2_R16_KLD}} vs full v2 |
| `Inkling-2-Mini-Unbound-LoRA-r64.gguf` / `-r16.gguf` | v1.0 (legacy) | see the v1.0 revision |

```bash
llama-server -m Inkling-2-Mini-Q8_0.gguf --lora Inkling-2-Mini-Unbound-v2.1-LoRA-r64.gguf \
    -ngl 99 -c 32768 --jinja --temp 0.6 --top-p 0.95
```

They are rank-64 / rank-16 SVD fits of the exact difference between v2.1 and the original in the 120 changed
attention tensors, applied at scale 1 on ggml-org's
[Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF). Not bit-exact; for exact weights use the
full GGUFs above.

## Known limits

- **Drift.** KLD {{KLD_V21}} from the original, the most of any release we measured (AtomicChat's LoRA 0.0078).
  MMLU-300 and GSM8K-100 stay within noise, but expect more wording differences on ordinary requests.
- **Loops at greedy decoding.** With thinking on and greedy decoding, {{V21_CAP_E09}} of harmful prompts hit the
  4,096-token thinking cap. Use temperature 0.6 / top_p 0.95 and set a cap.
- **Not every answer is useful.** About 15–16% of harmful requests still get an answer the judge rates as not
  fulfilling the request.
- **Crisis handling.** It was not trained to add crisis resources and may leave them out. Deployments that can reach
  people at risk need their own handling.
- **Not yet measured:** the full capability suite v1.0 passed, long-context, vision and tool use.
- Full list: [main model card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound#known-limits).

## Version history

| Version | Files in this repo | Useful off / on ({{N_BASIS}} prompts) | KLD |
|---|---|---|---|
| **v2.1** (current) | `Inkling-2-Mini-Unbound-v2.1-<TYPE>.gguf` | **{{V21_OFF}} / {{V21_E09}}** | {{KLD_V21}} |
| v2 | `Inkling-2-Mini-Unbound-v2-<TYPE>.gguf`, also the [`v2` branch](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF/tree/v2) | {{V2_OFF}} / {{V2_E09}} | 0.0318 |
| v1.0 | `Inkling-2-Mini-Unbound-<TYPE>.gguf` (no version in the name), also the [`v1.0` revision](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF/tree/v1.0) | {{V1_OFF}} / {{V1_E09}} | 0.0102 |

v2 loops less with greedy decoding; v1.0 is the smallest change from the original (weight edit only, 7 MB LoRA).

## Intended use and misuse

Intended for research on refusal and model editing, red-teaming and safety evaluation, and for developers who apply
their own content policy in a system they control. Not intended for products that serve the public without
moderation, content that facilitates violence, weapons capable of mass harm, attacks on computer systems or abuse of
children or other people, or anything illegal where you use it. The base model's
[Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) applies.

## Licence

Apache 2.0, like the base model, Inkling 2 Mini by Thinking Machines Lab, Inc. 120 attention tensors differ from the
original; every other tensor is ggml-org's conversion of the original, unmodified before quantization.

## About nokeep

This model was built, and its compute paid for, by [nokeep](https://nokeep.ai/?ref=huggingface). nokeep is a private
AI app for uncensored chat, images and video. If this model is useful to you, the best way to support more releases
like it is to [try nokeep](https://nokeep.ai/?ref=huggingface).
