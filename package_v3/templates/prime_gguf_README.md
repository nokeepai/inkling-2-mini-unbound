---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model: {{names.prime_repo}}
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags:
- gguf
- llama.cpp
- inkling2
- reasoning
---

![Inkling-2-Mini-Prime](https://huggingface.co/{{names.prime_repo}}/resolve/main/charts/banner_prime_v3.png)

# {{names.prime_gguf_title}}

**Inkling 2 Mini that stops over-thinking and gets on with the job, packaged to run on your own computer.**

[{{names.prime_title}}](https://huggingface.co/{{names.prime_repo}}) is for **agents and multi-turn work**, where it
finishes multi-step debugging instead of stalling; for **long system prompts**, where it follows every rule more
often; and for anyone tired of waiting: it thinks {{headline.think_reduction}} as much as the original, so answers
come faster. It is a drop-in replacement for [Inkling 2 Mini](https://huggingface.co/{{names.stock_repo}}), Thinking
Machines Lab's open-weights reasoning model, which takes text and images and "thinks" in private before it answers.
**It is not uncensored: it refuses what the original refuses.** (If you want the uncensored model with the same
Prime add-on, that is [{{names.unbound_repo}}](https://huggingface.co/{{names.unbound_repo}}).)

{{names.prime_name}} is the original model plus one small extra training step, the **Prime add-on**: a small set of
weight changes (a LoRA adapter) trained only on harmless tasks. This repo holds {{names.prime_name}} as GGUF files:
the format that llama.cpp, LM Studio and Ollama run on an ordinary computer.

Built and paid for by [nokeep](https://nokeep.ai/?ref=huggingface).

![{{names.prime_name}} against the original Inkling 2 Mini: hard debugging tasks solved, turns with no code written, and tokens used per task](https://huggingface.co/{{names.prime_repo}}/resolve/main/charts/hero_v3.png)

What the Prime add-on does, measured on the full-precision weights (details on the
[main {{names.prime_name}} card](https://huggingface.co/{{names.prime_repo}})):

- **It finishes multi-step work.** On {{prime_vs_stock.debug_n_tasks}} hard debugging tasks it solved
  {{prime_vs_stock.debug_solved_prime}}, against {{prime_vs_stock.debug_solved_stock}} for the original model, and
  its turns with no code written fell from {{prime_vs_stock.debug_nocode_stock}} to
  {{prime_vs_stock.debug_nocode_prime}}. (A hard debugging task is a small program with three interacting bugs.
  The model sees the failing test output, proposes a fix, sees the new output, and repeats for up to 12 turns. A
  "turn with no code written" is a turn it spent only thinking.)
- **It follows long system prompts better:** every rule followed in {{sysprompt.kept_fix}} of long-system-prompt
  tasks, against {{sysprompt.kept_stock}} for the original.
- **It thinks less:** {{headline.think_reduction}} as much as the original on everyday prompts, and seconds per
  everyday answer went from {{speed.sec_main_stock}} to {{speed.sec_main_prime}}.
- **No accuracy gain is claimed.** On public coding, maths and knowledge benchmarks it scores level with the
  original, within {{indep.margin_pts}}.

Numbers in brackets are 95% intervals: the range the true value plausibly sits in, given how many items were tested.

## Which file should I pick?

Each file is the same model at a different level of **quantisation**: the weights are stored with fewer bits, so
the file is smaller and runs on less memory, at some cost in quality. Pick the largest file that fits your
graphics card's memory (VRAM). You need about the file size, plus a few GB for the conversation.

| File | Type | Who it is for |
|---|---|---|
| `{{names.prime_title}}-BF16.gguf` | BF16, full precision | Making your own quantised files. Too large for most computers. |
| `{{names.prime_title}}-Q8_0.gguf` | Q8_0 | 32 GB+ of VRAM, or plenty of system RAM. Near lossless. |
| `{{names.prime_title}}-Q6_K.gguf` | Q6_K | 24 to 32 GB of VRAM. |
| `{{names.prime_title}}-Q5_K_M.gguf` | Q5_K_M | 24 to 32 GB of VRAM. |
| **`{{names.prime_title}}-Q4_K_M.gguf`** | **Q4_K_M** | **Most people: 16 to 24 GB of VRAM.** |
| `{{names.prime_title}}-IQ3_M.gguf` | IQ3_M | 12 to 16 GB of VRAM. Noticeably lossy. |
| `{{names.prime_title}}-IQ2_M.gguf` | IQ2_M | Under 12 GB of VRAM. Lossy. |
| `mmproj-Inkling-2-Mini-Q8_0.gguf` | vision projector | Needed only if you want to send images. Use this one. |
| `mmproj-Inkling-2-Mini-BF16.gguf` | vision projector, full precision | The same, unquantised. |

**What are the `mmproj` files?** Inkling 2 Mini reads images through a separate small component, the vision
projector, that turns a picture into something the language model can read. llama.cpp keeps it in its own file.
The Prime add-on did not change it, so these are the original model's projector files. Skip them if you only use text.

**If the model does not fit.** Inkling 2 Mini is a mixture-of-experts model: only part of it is used for each word.
llama.cpp can keep some of those experts in ordinary system RAM with `--n-cpu-moe N`. That works, just slower.

File sizes and SHA-256 checksums are in `SHA256SUMS` and `MANIFEST.tsv` in this repo, and Hugging Face shows both
for each file.

## How to run it

You need a llama.cpp build with Inkling 2 support that includes the loop fix from PR #32704. We used commit
`19bc41dbb`; tagged releases up to b12608 do not have it.

**llama.cpp**

```bash
hf download {{names.prime_gguf_repo}} {{names.prime_gguf_file}} mmproj-Inkling-2-Mini-Q8_0.gguf --local-dir models
./build/bin/llama-server -m models/{{names.prime_gguf_file}} --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf \
    --jinja -c 32768 -ngl 99 --temp {{settings.temperature}} --top-p {{settings.top_p}}
```

Then pass `"chat_template_kwargs": {"reasoning_effort": {{settings.effort}}}` with each request
(`reasoning_effort` sets how much the model thinks before answering; see [Tested settings](#tested-settings)). llama.cpp's own
default top_p is 0.95, so set it explicitly as above.

**LM Studio.** Search for `{{names.prime_gguf_repo}}`, pick a file from the table above, and set temperature
{{settings.temperature}} and top_p {{settings.top_p}}.

**Ollama.**

```bash
cat > Modelfile <<'EOF'
FROM ./models/{{names.prime_gguf_file}}
PARAMETER temperature {{settings.temperature}}
PARAMETER top_p {{settings.top_p}}
EOF
ollama create inkling-prime -f Modelfile
ollama run inkling-prime
```

## Tested settings

Every number on this card and on the main card was measured with these settings, so they are the ones we recommend.

| Setting | Value | What it does |
|---|---|---|
| `reasoning_effort` | {{settings.effort}} | How much the model thinks before it answers. Passed through the chat template. |
| `temperature` | {{settings.temperature}} | Randomness of the wording. |
| `top_p` | {{settings.top_p}} | Which candidate words are allowed. llama.cpp defaults to 0.95, so set it yourself. |
| Thinking limit | {{settings.think_cap}} tokens | A cap on the private thinking; at the cap the model has to answer. |

## Does the Prime add-on still work after quantisation?

Quantising can blur a small change to the weights, so we re-ran the hard debugging tasks (the same
{{gguf.n_tasks}} as on the main card) on these files with llama.cpp. For comparison, the original Inkling 2 Mini
was quantised to Q4_K_M in exactly the same way and run the same way.

| File | Debugging tasks solved | Turns with no code written | Tokens per task | Maths word problems correct | Coding problems passed |
|---|---|---|---|---|---|
| {{names.prime_name}} Q8_0 | {{gguf.solve_prime_q8}} | {{gguf.nocode_prime_q8}} | {{gguf.tokens_prime_q8}} | {{gguf.gsm8k_number_prime_q8}} | {{gguf.humaneval_prime_q8}} |
| **{{names.prime_name}} Q4_K_M** | **{{gguf.solve_prime_q4km}}** | **{{gguf.nocode_prime_q4km}}** | **{{gguf.tokens_prime_q4km}}** | **{{gguf.gsm8k_number_prime_q4km}}** | **{{gguf.humaneval_prime_q4km}}** |
| {{names.prime_name}} IQ2_M | {{gguf.solve_prime_iq2m}} | {{gguf.nocode_prime_iq2m}} | {{gguf.tokens_prime_iq2m}} | {{gguf.gsm8k_number_prime_iq2m}} | {{gguf.humaneval_prime_iq2m}} |
| Original model, Q4_K_M (made the same way) | {{gguf.solve_stock_q4km}} | {{gguf.nocode_stock_q4km}} | {{gguf.tokens_stock_q4km}} | {{gguf.gsm8k_number_stock_q4km}} | {{gguf.humaneval_stock_q4km}} |

What each column means:

- **Debugging tasks solved:** share of the {{gguf.n_tasks}} hard debugging tasks where all tests pass within 12 turns.
- **Turns with no code written:** share of turns where the model only thought and proposed no fix. Lower is better.
- **Tokens per task:** the average amount of text (thinking plus answers) the model wrote per task. A token is
  roughly three quarters of a word. Lower means faster.
- **Maths word problems correct:** the first 100 problems of GSM8K, a public set of grade-school maths word
  problems, counting the final number as correct even if written with units.
- **Coding problems passed:** the first 50 problems of HumanEval, a public set of short Python functions checked by
  tests.

The short answer: mostly yes. Q8_0 and Q4_K_M keep the Prime add-on's effect fully. The smallest file, IQ2_M, still solves
{{gguf.solve_prime_iq2m}} of the tasks but stalls on {{gguf.nocode_prime_iq2m}} of turns. The original model
quantised the same way stalls far more. Each file was run once. Q6_K, Q5_K_M and IQ3_M were not re-run.

<details>
<summary>How these files were made</summary>

- **Patched, not re-converted.** [`scripts/v3stack/gguf_v3.py`](https://github.com/{{names.github_repo}}/blob/main/scripts/v3stack/gguf_v3.py) reads the merged {{names.prime_name}} checkpoint
  and writes its changed attention and expert tensors into ggml-org's official `Inkling-2-Mini-BF16.gguf`. Before
  patching, it proves the checkpoint-to-GGUF tensor mapping byte-exact on all 192 mapped tensors (144 attention,
  48 expert) using the original checkpoint and the official file; `patch` refuses to run without that proof.
- **Verified.** After patching, every mapped tensor must equal the merged checkpoint pushed through the proven
  mapping, every other tensor must be byte-identical to the official file, and only `general.name` changes in the
  header.
- **Quantised.** llama.cpp `llama-quantize` at commit `19bc41dbb`: Q8_0 directly; Q6_K, Q5_K_M, Q4_K_M, IQ3_M and
  IQ2_M with ggml-org's `imatrix.gguf` (an importance matrix: a record of which weights matter most, used to
  quantise more carefully) and `attn_q`, `attn_k`, `attn_v`, `attn_output` and `attn_gate` kept at
  q8_0 ([`scripts/v3stack/gguf_v3.sh`](https://github.com/{{names.github_repo}}/blob/main/scripts/v3stack/gguf_v3.sh)).
- The scripts are in the [GitHub repo](https://github.com/{{names.github_repo}}).

</details>

## Limits

- **No accuracy gain.** It thinks less and finishes more; it is not smarter. One possible small coding dip is not
  confirmed: MBPP, a set of short Python problems, came out {{pool.mbpp_prime_diff}} over {{pool.mbpp_prime_seeds}}
  runs with different random seeds (a seed fixes the random choices the model makes while writing).
- On sensitive requests it answers as often as the original and slightly less fully (delivery score
  {{ubench.delivery_stock}} → {{ubench.delivery_prime}}).
- **The quantisation check is small:** one run per file, three of the seven files, {{gguf.n_tasks}} debugging tasks.
- **Not measured:** long documents, images and tool calling. Only English was tested.
- The full evaluation and limits are on the [main {{names.prime_name}} card](https://huggingface.co/{{names.prime_repo}}).
  Everything the [original model's card](https://huggingface.co/{{names.stock_repo}}) says about capabilities and
  limits still applies.

## Licence and use

Apache 2.0, like the base model, Inkling 2 Mini by Thinking Machines Lab, Inc. The base model's
[Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) applies.
