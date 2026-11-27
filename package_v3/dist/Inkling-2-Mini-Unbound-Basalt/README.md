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

![Inkling-2-Mini-Unbound v3](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/banner_v3.png)

# Inkling-2-Mini-Unbound Basalt

**An uncensored Inkling 2 Mini in one file, for NVIDIA Blackwell GPUs.**

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is Thinking Machines Lab's open-weights reasoning
model. It takes text and images, and it "thinks" in private before it answers. Out of the box it refuses most
sensitive requests. [Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) is the same model with
the refusals removed.

**This repo holds Unbound packaged for [Basalt](https://github.com/jesdga95/basalt)**, an inference engine built for
NVIDIA's Blackwell GPUs. Everything Basalt needs (weights, tokenizer, chat template, vision part and a speed-up
helper) is in a single file: `Inkling-2-Mini-Unbound-v3-NVFP4-q8.basalt` (19,734,300,585 bytes). The weights are
stored mostly in NVFP4, a compact 4-bit number format that Blackwell GPUs compute with directly.

It is built by [nokeep](https://nokeep.ai/?ref=huggingface) for people who apply their own content rules:
researchers, red-teamers, fiction writers, and developers running a system they control.

Current version: **v3**, released 2026-11-27. Files from earlier versions are still here
(see [Version history](#version-history)). The [main model card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) has
the full evaluation.

> **This model follows harmful instructions.** It gives detailed answers to requests for harmful, dangerous or
> illegal content that the original model declines. It has no safety training layered back on. Do not deploy it to
> end users without your own safeguards, and do not use it where its outputs can reach someone who did not choose to
> use an uncensored model.

## Is this the right download?

| You have | Get |
|---|---|
| An NVIDIA Blackwell GPU (RTX 50-series, RTX PRO 6000; compute capability 12.0) and want one file and one command | **this repo** |
| Any other computer, including Macs, older GPUs and CPUs, with llama.cpp, LM Studio or Ollama | [nokeepai/Inkling-2-Mini-Unbound-GGUF](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF) |
| A GPU server running Transformers, SGLang or vLLM | [nokeepai/Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) |

## How to run it

You need Basalt **0.5.5 or later**, an NVIDIA Blackwell GPU with CUDA 13, and a driver from the R580 branch or newer.

```sh
hf download nokeepai/Inkling-2-Mini-Unbound-Basalt Inkling-2-Mini-Unbound-v3-NVFP4-q8.basalt --local-dir models
bin/basalt serve models/Inkling-2-Mini-Unbound-v3-NVFP4-q8.basalt
```

The server listens on `127.0.0.1:8080` and accepts requests in the OpenAI and Anthropic API formats, so most chat
apps and client libraries can talk to it.

Flags, context length, images and the drafter (the built-in speed-up helper) are covered in Basalt's
[Inkling 2 Mini guide](https://github.com/jesdga95/basalt/blob/main/docs/models/inkling2.md).

## Settings

| Setting | Value | What it does |
|---|---|---|
| `temperature` | 0.6 | Randomness. Send it with each request. Do not use 0: earlier versions looped in their thinking at 0. |
| `top_p` | 1.0 | Send it with each request. |
| `reasoning_effort` | `none` to `max` | How much the model thinks before it answers, set per request. Even at `none` it writes a short thinking block, so leave room in `max_tokens`. |
| Thinking limit | 4096 tokens | The most private reasoning our tests let it write before it had to answer. A token is a word or part of a word. |

Our tests used `reasoning_effort` 0.9 through the model's chat template, on the full-precision
weights in the main repo, not on this file.

## What is new in v3

The previous version, v2.1, had one real flaw: **it thought too much.** It often kept reasoning until it hit the
thinking limit. On multi-step work, such as fixing a program over several rounds, it spent most turns thinking and
wrote no code at all. The original Inkling 2 Mini has the same habit.

v3 adds one small extra training step, the **Prime add-on**: a small set of weight changes (a LoRA adapter)
trained only on harmless tasks. It is the same add-on that turns the original Inkling 2 Mini into
Inkling-2-Mini-Prime. It teaches the model to stop deliberating and act. Because it was trained only on harmless
tasks (debugging, maths, writing, everyday questions), it changes how long the model thinks, not what it is willing
to say.

![Unbound v3 against v2.1: hard debugging tasks solved, turns with no code written, and tokens used per task](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/hero_v3.png)

| What we measured (full-precision weights, not this file) | v2.1 | **v3** |
|---|---|---|
| Hard debugging tasks solved (15 tasks) | 73.3% [48.0, 89.1] | **100.0% [79.6, 100.0]** |
| Debugging turns where it wrote no code | 88.7% | **0.0%** |
| Tokens used per debugging task | 26,130 | **2,429** |
| Sensitive requests: willing to answer | 99.4% [98.7, 99.7] | **99.9% [99.5, 100.0]** |

How to read the table:

- **Hard debugging tasks** are 15 small programs with three interacting bugs each. The
  model sees the failing test output, proposes a fix, sees the new output, and repeats for up to 12 turns.
- **A turn with no code written** is a round where the model spent its reply thinking and never proposed a fix.
- **Sensitive requests** are uncensoredbench, a benchmark of requests that models usually refuse.
- **Numbers in brackets** are 95% confidence intervals: the range the true value plausibly sits in, given how many
  items were tested.

In short, v3 uses about half the thinking, and time per everyday answer went from
161.7 s to 52.5 s, because it writes fewer tokens. The engine itself is no faster.

**What v3 is not.** It is not smarter: on public benchmarks it scores level with v2.1, within about
2 points, and we claim no accuracy gain. It is not more uncensored either: v2.1 already answered almost
everything. The [main card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) (under "Full results") lists every
benchmark.

Inkling-2-Mini-Prime is the original, still-censored model with the same Prime add-on:
[nokeepai/Inkling-2-Mini-Prime](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime).

## Version history

Each version fixed the biggest problem of the one before.

| Version | Date | What changed |
|---|---|---|
| v1.0 | 2026-11-21 | Refusals removed by a direct edit of the weights. No training. |
| v2 | 2026-11-22 | v1.0 stopped refusing, but many answers still dodged. v2 was trained on the model's own best direct answers. |
| v2.1 | 2026-11-23 | A second round of that training, so it holds back less. |
| **v3** (current) | 2026-11-27 | The Prime add-on: it stops over-thinking and finishes multi-step work. |

The older files are still in this repo:

| Version | File | Also at |
|---|---|---|
| **v3** | `Inkling-2-Mini-Unbound-v3-NVFP4-q8.basalt` | `main` |
| v2.1 | `Inkling-2-Mini-Unbound-v2.1-NVFP4-q8.basalt` | the [`v2.1` branch](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-Basalt/tree/v2.1) |
| v2 | `Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt` | the [`v2` branch](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-Basalt/tree/v2) |
| v1.0 | `Inkling-2-Mini-Unbound-NVFP4-q8.basalt` (no version in the name) | the [`v1.0` revision](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-Basalt/tree/v1.0) |

The full story of each release is on the [main card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) (under "Version history").

## What was tested on this file

- **Tested:** the file's contents, byte by byte (details below). Every value in the file was recomputed from the v3 weights and matched (Basalt converter
  v0.5.5). The checker's own summary: RESULT: VERIFY PASS, built with converter v0.5.5 (commit ff0b978d): every tensor re-derived from the merged checkpoint, metadata identical to the official 0.5.5 pack apart from the declared fields.
- **Not tested:** running this v3 file on a GPU (`basalt serve`). None of the scores on this page or on the main
  card come from this file.
- **Speed:** not re-measured. This file has the same layout, size class and number formats as the earlier Basalt
  files, so expect the same speed per token. The built-in drafter (a small helper model that guesses the next few
  tokens so the main model can confirm several at once) is the original model's, so its guesses may be accepted a
  little less often than in the official file.

<details>
<summary>How this file was made and checked</summary>

**Build.** The same layout as the official
[jesdga/Inkling-2-Mini-Basalt](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt) pack
(`Inkling-2-Mini-NVFP4-q8`), with v3's weights in it, built the way v2.1's pack was: Basalt v0.5.5's own recipe,
`tools/convert/convert.py --checkpoint`, run on the v3 safetensors. Chat template, tokenizer, vision tower and
drafter are the original release's. sha256 `63cfce164daa9c35dd111ed28275c6f0e2a7173eaed292c9ef1636a0360072e6`.

**Verification** ([`scripts/v3rel/verify_basalt_v3.py`](https://github.com/nokeepai/inkling-2-mini-unbound/blob/main/scripts/v3rel/verify_basalt_v3.py)). v2.1's checker required the experts to match the official
pack byte for byte; v3 changes the experts, so v3 has its own checker. It re-runs Basalt 0.5.5's converter
transform on the merged v3 checkpoint and requires every byte of this pack to equal either the official 0.5.5 pack
(`Inkling-2-Mini-NVFP4-DQ8.basalt`, pinned by revision and sha256) or that recomputation, with changes allowed only
where the v3 weights differ. In order: every section's SHA-256; same sections, with tokenizer and expert layout
identical to the official pack; every expert of every layer; every GGUF-section tensor (same table, changed Q8_0
tensors cross-checked with an independent Q8_0 implementation); vision tensors identical; metadata identical apart
from name and source fields.

</details>

## Limits

- **Extra rounding.** Storing the weights in NVFP4 and Q8_0 (an 8-bit format) loses a little precision compared with
  the full-precision weights the evaluations used. See Basalt's
  [precision notes](https://github.com/jesdga95/basalt/blob/main/docs/precision/inkling2.md).
- **Engine checks are old.** Checks inside Basalt itself (a serving sanity check, refusal and crisis probes) were
  done on the v1.0 file and have not been repeated for later versions.
- **Willing is not the same as useful.** Some answers to harmful requests still hold part of the request back.
- **No crisis handling.** It was not trained to add crisis resources. **Do not use this model where it can reach
  people at risk without your own crisis handling.**
- **Not measured:** long documents, images and tool calling. Only English was tested.
- The full list is on the [main card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) (under "Limits").

## Intended use

For research on refusal and model editing, red-teaming and safety evaluation, fiction, and developers who apply
their own content policy in a system they control. Not for products that serve the public without moderation, and
not for content that facilitates violence, weapons capable of mass harm, attacks on computer systems, or the abuse
of children or anyone else, or for anything illegal where you use it. The base model's
[Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) applies.

## Licence

Apache 2.0, like the base model, Inkling 2 Mini by Thinking Machines Lab, Inc. Basalt is by its own authors under
its own licence; this repo holds only the model file.

## About nokeep

Built and paid for by [nokeep](https://nokeep.ai/?ref=huggingface): private, uncensored AI chat, images and video.
