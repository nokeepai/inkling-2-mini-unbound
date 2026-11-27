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

![Inkling-2-Mini-Unbound v3](charts/banner_v3.png)

# Inkling-2-Mini-Unbound GGUF

**Run an uncensored Inkling 2 Mini on your own computer, with llama.cpp, LM Studio or Ollama.**

[Inkling 2 Mini](https://huggingface.co/{{names.stock_repo}}) is Thinking Machines Lab's open-weights reasoning
model. It takes text and images, and it "thinks" in private before it answers. Out of the box it refuses most
sensitive requests. [Inkling-2-Mini-Unbound](https://huggingface.co/{{names.unbound_repo}}) is the same model with
the refusals removed. **This repo holds Unbound as GGUF files**: the single-file format that llama.cpp, LM Studio and
Ollama load, so you can run it on a desktop GPU, a Mac or a CPU.

It is built by [nokeep](https://nokeep.ai/?ref=huggingface) for people who apply their own content rules:
researchers, red-teamers, fiction writers, and developers running a system they control.

Current version: **{{names.version}}**, released {{names.release_date}}. Files from earlier versions are still here
(see [Version history](#version-history)). The [main model card](https://huggingface.co/{{names.unbound_repo}}) has
the full evaluation.

> **This model follows harmful instructions.** It gives detailed answers to requests for harmful, dangerous or
> illegal content that the original model declines. It has no safety training layered back on. Do not deploy it to
> end users without your own safeguards.

## Which file do I need?

Each file is the whole model at a different level of **quantisation**: the weights are stored with fewer bits, so
the file is smaller and needs less memory, at the cost of a little accuracy. Smaller files lose more.

**Most people: `Q4_K_M`.** Pick a smaller file only if Q4_K_M does not fit.

| File | Size | Fits fully on a GPU with | Quality |
|---|---|---|---|
| `Inkling-2-Mini-Unbound-v3-Q8_0.gguf` | about 28 GB | 32 GB or more | Near the full model |
| `Inkling-2-Mini-Unbound-v3-Q6_K.gguf` | about 24 GB | 24 to 32 GB | Close to Q8_0 |
| `Inkling-2-Mini-Unbound-v3-Q5_K_M.gguf` | about 20 GB | 24 to 32 GB | Close to Q8_0 |
| **`Inkling-2-Mini-Unbound-v3-Q4_K_M.gguf`** | **about 18 GB** | **16 to 24 GB** | **Recommended** |
| `Inkling-2-Mini-Unbound-v3-IQ3_M.gguf` | about 13 GB | 12 to 16 GB | Noticeably lossy |
| `Inkling-2-Mini-Unbound-v3-IQ2_M.gguf` | about 11 GB | under 12 GB | Lossy |
| `Inkling-2-Mini-Unbound-v3-BF16.gguf` | about 52 GB | (not for running) | Unquantised; for making your own quants |
| `mmproj-Inkling-2-Mini-Q8_0.gguf` (or `-BF16`) | under 1 GB | add to any file above | Needed only for image input |

How to read this:

- **"Fits fully on a GPU"** means video memory (VRAM). You need about the file size plus a few GB for the
  conversation. If it does not fit, llama.cpp can keep part of the model in ordinary RAM with `--n-cpu-moe N`. This
  works because Inkling 2 Mini is a mixture-of-experts model (only some of its weights are used for each word). It is
  slower, but it runs. With that option, Q4_K_M is still a good choice under 12 GB of VRAM.
- **mmproj** is the vision projector: the small extra file that lets the model read images. It is the original
  model's, unchanged. Skip it if you only use text.
- Exact sizes and SHA-256 checksums are in `MANIFEST.tsv` in this repo, and Hugging Face shows both for each file.

## How to run it

### llama.cpp

You need a llama.cpp build with Inkling 2 support that includes the loop fix from PR #32704. We used commit
`19bc41dbb`; tagged releases up to b12608 do not have it.

```bash
hf download {{names.unbound_repo}}-GGUF \
    Inkling-2-Mini-Unbound-v3-Q4_K_M.gguf mmproj-Inkling-2-Mini-Q8_0.gguf --local-dir models

./build/bin/llama-server \
    -m models/Inkling-2-Mini-Unbound-v3-Q4_K_M.gguf \
    --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf \
    --jinja -c 32768 -ngl 99 --temp {{settings.temperature}} --top-p {{settings.top_p}}
```

### LM Studio

Search for `{{names.unbound_repo}}-GGUF` and pick a file with `-v3` in its name (Q4_K_M unless the table above says
otherwise). Set temperature to {{settings.temperature}} and top_p to {{settings.top_p}}.

### Ollama

```bash
cat > Modelfile <<'EOF'
FROM ./models/Inkling-2-Mini-Unbound-v3-Q4_K_M.gguf
PARAMETER temperature {{settings.temperature}}
PARAMETER top_p {{settings.top_p}}
EOF
ollama create inkling-unbound-v3 -f Modelfile
ollama run inkling-unbound-v3
```

## Settings

These are the settings all our tests used, and the only ones we recommend:

| Setting | Value | What it does |
|---|---|---|
| `reasoning_effort` | {{settings.effort}} | How much the model thinks before it answers. It is passed through the chat template. |
| `temperature` | {{settings.temperature}} | Randomness. Do not use 0: earlier versions looped in their thinking at 0. |
| `top_p` | {{settings.top_p}} | llama.cpp defaults to 0.95, so set it yourself. |
| Thinking limit | {{settings.think_cap}} tokens | The most private reasoning we let it write before it must answer. A token is a word or part of a word. |

The main evaluation ran on the full-precision weights in the main repo, not on these files. The section
[Does the Prime add-on still work in these files?](#does-the-prime-add-on-still-work-in-these-files) shows what we checked on
the files themselves.

## What is new in v3

The previous version, v2.1, had one real flaw: **it thought too much.** It often kept reasoning until it hit the
thinking limit. On multi-step work, such as fixing a program over several rounds, it spent most turns thinking and
wrote no code at all. The original Inkling 2 Mini has the same habit.

v3 adds one small extra training step, the **Prime add-on**: a small set of weight changes (a LoRA adapter)
trained only on harmless tasks. It is the same add-on that turns the original Inkling 2 Mini into
Inkling-2-Mini-Prime. It teaches the model to stop deliberating and act. Because it was trained only on harmless
tasks (debugging, maths, writing, everyday questions), it changes how long the model thinks, not what it is willing
to say.

![Unbound v3 against v2.1: hard debugging tasks solved, turns with no code written, and tokens used per task](charts/hero_v3.png)

| What we measured (full-precision weights) | v2.1 | **v3** |
|---|---|---|
| Hard debugging tasks solved ({{v3_vs_v21.debug_n_tasks}} tasks) | {{v3_vs_v21.debug_solved_v21}} | **{{v3_vs_v21.debug_solved_v3}}** |
| Debugging turns where it wrote no code | {{v3_vs_v21.debug_nocode_v21}} | **{{v3_vs_v21.debug_nocode_v3}}** |
| Tokens used per debugging task | {{v3_vs_v21.debug_tokens_v21}} | **{{v3_vs_v21.debug_tokens_v3}}** |
| Sensitive requests: answers cut off by the thinking limit | {{speed.cap_ubench_v21}} | **{{speed.cap_ubench_v3}}** |
| Sensitive requests: willing to answer | {{ubench.willing_v21}} | **{{ubench.willing_v3}}** |

How to read the table:

- **Hard debugging tasks** are {{v3_vs_v21.debug_n_tasks}} small programs with three interacting bugs each. The
  model sees the failing test output, proposes a fix, sees the new output, and repeats for up to 12 turns.
- **A turn with no code written** is a round where the model spent its reply thinking and never proposed a fix.
- **Sensitive requests** are uncensoredbench, a benchmark of requests that models usually refuse.
- **Numbers in brackets** are 95% confidence intervals: the range the true value plausibly sits in, given how many
  items were tested.

In short, v3 uses {{headline.think_reduction}} the thinking, and time per everyday answer went from
{{speed.sec_main_v21}} to {{speed.sec_main_v3}}, because it writes fewer tokens.

**What v3 is not.** It is not smarter: on public benchmarks it scores level with v2.1, within about
{{indep.margin_pts}}, and we claim no accuracy gain. It is not more uncensored either: v2.1 already answered almost
everything. The [main card](https://huggingface.co/{{names.unbound_repo}}) (under "Full results") lists every
benchmark, including the few that scored slightly lower.

Inkling-2-Mini-Prime is the original, still-censored model with the same Prime add-on:
[{{names.prime_repo}}](https://huggingface.co/{{names.prime_repo}}).

## Version history

Each version fixed the biggest problem of the one before.

| Version | Date | What changed |
|---|---|---|
| v1.0 | 2026-11-21 | Refusals removed by a direct edit of the weights. No training. |
| v2 | 2026-11-22 | v1.0 stopped refusing, but many answers still dodged. v2 was trained on the model's own best direct answers. |
| v2.1 | 2026-11-23 | A second round of that training, so it holds back less. |
| **v3** (current) | {{names.release_date}} | The Prime add-on: it stops over-thinking and finishes multi-step work. |

The older files are still in this repo. You can tell them apart by the name:

| Version | File names | Also at |
|---|---|---|
| **v3** | `Inkling-2-Mini-Unbound-v3-<TYPE>.gguf` | `main` |
| v2.1 | `Inkling-2-Mini-Unbound-v2.1-<TYPE>.gguf` | the [`v2.1` branch](https://huggingface.co/{{names.unbound_repo}}-GGUF/tree/v2.1) |
| v2 | `Inkling-2-Mini-Unbound-v2-<TYPE>.gguf` | the [`v2` branch](https://huggingface.co/{{names.unbound_repo}}-GGUF/tree/v2) |
| v1.0 | `Inkling-2-Mini-Unbound-<TYPE>.gguf` (no version in the name) | the [`v1.0` revision](https://huggingface.co/{{names.unbound_repo}}-GGUF/tree/v1.0) |

`<TYPE>` is the quantisation, such as `Q4_K_M`. The full story of each release is on the
[main card](https://huggingface.co/{{names.unbound_repo}}) (under "Version history").

## Does the Prime add-on still work in these files?

Yes. Quantising can change how a model behaves, so we re-ran the hard debugging tasks and two quick accuracy checks
on these files with llama.cpp ({{gguf.n_tasks}} debugging tasks, the same ones as above).

| File | Hard debugging solved | Turns with no code | Tokens per task (mean) | GSM8K, first 100 | HumanEval, first 50 |
|---|---|---|---|---|---|
| **Unbound v3, Q4_K_M** | **{{gguf.solve_unbound_q4km}}** | **{{gguf.nocode_unbound_q4km}}** | **{{gguf.tokens_unbound_q4km}}** | **{{gguf.gsm8k_number_unbound_q4km}}** | **{{gguf.humaneval_unbound_q4km}}** |
| Unbound v3, IQ2_M (the smallest file) | {{gguf.solve_unbound_iq2m}} | {{gguf.nocode_unbound_iq2m}} | {{gguf.tokens_unbound_iq2m}} | {{gguf.gsm8k_number_unbound_iq2m}} | {{gguf.humaneval_unbound_iq2m}} |
| Original Inkling 2 Mini, Q4_K_M (no Prime add-on, made the same way) | {{gguf.solve_stock_q4km}} | {{gguf.nocode_stock_q4km}} | {{gguf.tokens_stock_q4km}} | {{gguf.gsm8k_number_stock_q4km}} | {{gguf.humaneval_stock_q4km}} |

What each column means:

- **Hard debugging solved:** share of the debugging tasks where the model got every test passing within 12 turns.
- **Turns with no code:** share of turns where it only thought and proposed no fix. Lower is better.
- **Tokens per task:** everything the model wrote, thinking included, averaged over the tasks. Lower means faster.
- **GSM8K, first 100:** the first 100 school maths word problems of the GSM8K benchmark; the share with the right
  final number.
- **HumanEval, first 50:** the first 50 problems of the HumanEval coding benchmark, where the model writes a Python
  function that must pass hidden tests.

We did not run v2.1 as a GGUF file, so the comparison row is the original model, which stalls the same way v2.1
did on a GPU server. Even the smallest file, IQ2_M, keeps most of the Prime add-on's effect: it solves {{gguf.solve_unbound_iq2m}} of the
tasks but stalls on {{gguf.nocode_unbound_iq2m}} of turns.

<details>
<summary>How these files were made</summary>

- **Patched, not re-converted.** [`scripts/v3stack/gguf_v3.py`](https://github.com/{{names.github_repo}}/blob/main/scripts/v3stack/gguf_v3.py) reads the merged v3 checkpoint and writes its changed
  attention and expert tensors into ggml-org's official `Inkling-2-Mini-BF16.gguf`. Before patching, it proves the
  checkpoint-to-GGUF tensor mapping byte-exact on all 192 mapped tensors (144 attention, 48 expert) using the
  original checkpoint and the official file; `patch` refuses to run without that proof.
- **Verified.** After patching, every mapped tensor must equal the merged checkpoint pushed through the proven
  mapping, every other tensor must be byte-identical to the official file, and only `general.name` changes in the
  header.
- **Quantised.** llama.cpp `llama-quantize` at commit `19bc41dbb`. Q8_0 was made directly. Q6_K, Q5_K_M, Q4_K_M,
  IQ3_M and IQ2_M used ggml-org's `imatrix.gguf` (an importance matrix: a record of which weights matter most, so
  those are kept more precise), with `attn_q`, `attn_k`, `attn_v`, `attn_output` and `attn_gate` kept at q8_0
  ([`scripts/v3stack/gguf_v3.sh`](https://github.com/{{names.github_repo}}/blob/main/scripts/v3stack/gguf_v3.sh)).
- Sizes and SHA-256 sums: `MANIFEST.tsv` in this repo (file, bytes, sha256).

</details>

## Limits

- **Settings were tested on the full-precision weights.** The quantised files were checked only with the tests in
  the table above.
- **Willing is not the same as useful.** Some answers to harmful requests still hold part of the request back.
- **Slightly further from the original model than v2.1.** "Drift" measures how far the model's word-by-word
  predictions on ordinary text have moved from the original's (KL divergence; smaller is closer): v3
  {{drift.kld_v3}}, v2.1 {{drift.kld_v21}}.
- **Not measured:** long documents, images and tool calling. Only English was tested.
- **No crisis handling.** It was not trained to add crisis resources and may leave them out.
- The full list is on the [main card](https://huggingface.co/{{names.unbound_repo}}) (under "Limits").

## Intended use

For research on refusal and model editing, red-teaming and safety evaluation, fiction, and developers who apply
their own content policy in a system they control. Not for products that serve the public without moderation, and
not for content that facilitates violence, weapons capable of mass harm, attacks on computer systems, or the abuse
of children or anyone else, or for anything illegal where you use it. The base model's
[Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) applies.

## Licence

Apache 2.0, like the base model, Inkling 2 Mini by Thinking Machines Lab, Inc.

## About nokeep

Built and paid for by [nokeep](https://nokeep.ai/?ref=huggingface): private, uncensored AI chat, images and video.
