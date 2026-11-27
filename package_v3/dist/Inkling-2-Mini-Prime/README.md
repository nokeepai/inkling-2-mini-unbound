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
- finetune
- reasoning
- moe
---

![Inkling-2-Mini-Prime](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime/resolve/main/charts/banner_prime_v3.png)

# Inkling-2-Mini-Prime

**The original Inkling 2 Mini, trained to stop over-thinking: it finishes multi-step work instead of stalling and
thinks about half as much. Same refusals, same files: a drop-in replacement.**

Prime is for **agents and multi-turn work**, where it finishes multi-step debugging instead of
stalling; for **long system prompts**, where it follows every rule more often; and for anyone tired of waiting on
the thinking: it thinks about half as much, so answers come faster. It is a drop-in replacement
for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Thinking Machines Lab's open-weights reasoning
model: same architecture, files and chat template, so anything that runs Inkling 2 Mini runs Prime.
It is **not uncensored**: it refuses what the original refuses.

Prime is the original model plus one small extra training step, the **Prime add-on**: a small set
of weight changes (a LoRA adapter) trained only on harmless tasks. Nothing else was changed.

Built by [nokeep](https://nokeep.ai/?ref=huggingface). Released 2026-11-27.

## Headline numbers

- **Hard debugging tasks solved:** 80% → **100%**
  (15 tasks, original → Prime).
- **Debugging turns with no code written:** 86.2% → **0.0%**.
- **Long system prompts, every rule followed:** 112/160 (70%) → **148/160 (92.5%)**
  (40 tasks, run 4 times).
- **Thinking per everyday prompt:** 318 → **174** tokens (median);
  seconds per answer 119.9 s → **49 s**.
- **No accuracy gain is claimed:** level with the original on public coding, maths and knowledge benchmarks.

## Quick start

**On your own computer** (llama.cpp, LM Studio, Ollama): use the
[GGUF files](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime-GGUF). Most people want `Q4_K_M` (about 18 GB).

```bash
hf download nokeepai/Inkling-2-Mini-Prime-GGUF \
    Inkling-2-Mini-Prime-Q4_K_M.gguf mmproj-Inkling-2-Mini-Q8_0.gguf --local-dir models

llama-server -m models/Inkling-2-Mini-Prime-Q4_K_M.gguf \
    --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf \
    --jinja -c 32768 -ngl 99 --temp 0.6 --top-p 1.0
```

The [GGUF card](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime-GGUF) lists every file size, the llama.cpp version you
need, and the LM Studio and Ollama steps.

**On a GPU server** (Transformers, SGLang, vLLM): the weights in this repo load exactly like
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini); follow that card's instructions and swap in
`nokeepai/Inkling-2-Mini-Prime`. We ran it on one RTX 5090 (32 GB). Then call it like any OpenAI-compatible server:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "nokeepai/Inkling-2-Mini-Prime",
  "messages": [{"role": "user", "content": "Explain how a hash table handles collisions."}],
  "temperature": 0.6, "top_p": 1.0,
  "chat_template_kwargs": {"reasoning_effort": 0.9}
}'
```

**Settings.** Every number on this page was measured with these, so they are the ones we recommend. Limits are
counted in tokens, the word-pieces a model reads and writes:

| Setting | Value | What it does |
|---|---|---|
| `reasoning_effort` | 0.9 | How much the model thinks before it answers (0 turns thinking off; we tested 0.3 to 0.9). |
| `temperature` | 0.6 | Randomness. Do not use 0. |
| `top_p` | 1.0 | llama.cpp defaults to 0.95, so set it yourself. |
| Thinking limit | 4096 tokens | A cap on private thinking. If the model reaches it, it is made to stop and answer. |
| Answer limit | 8192 tokens | A cap on the visible answer. |

A lower effort, 0.3, was also measured: see "Can I just turn `reasoning_effort` down instead?"
under "Full results" below. No other values were tested.

## What the Prime add-on changes

The Prime add-on teaches the model to stop deliberating and act. It was trained only on harmless tasks (debugging,
maths, writing, everyday questions), so it changes how long the model thinks, not what it is willing to say.

The gain is largest on work done in several steps. The original often spends turn after turn thinking without
writing anything; Prime gets on with it, and uses far fewer tokens doing so.

![Prime against the original Inkling 2 Mini: hard debugging tasks solved, turns with no code written, and tokens used per task](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime/resolve/main/charts/hero_v3.png)

| What we measured | Original | **Prime** |
|---|---|---|
| Hard debugging tasks solved (15 tasks) | 80.0% [54.8, 93.0] | **100.0% [79.6, 100.0]** |
| Debugging turns where it wrote no code | 86.2% | **0.0%** |
| Tokens used per debugging task | 23,920 | **2,766** |
| Thinking per everyday prompt (median tokens) | 318 | **174** |
| Seconds per everyday answer, same server | 119.9 s | **49.0 s** |
| Long system prompts: every rule followed | 112/160 (70.0%) | **148/160 (92.5%)** |
| Sensitive requests: willing to answer | 64.5% [61.6, 67.2] | **65.3% [62.4, 68.0]** |

**How to read it.** A hard debugging task is a small program with three interacting bugs, which the model gets up
to 12 turns to fix. A **stalled turn** is a turn spent only thinking, with no code written. The long-system-prompt test is
40 tasks, each run 4 times.

**What Prime is not.** It is not smarter: on public benchmarks it scores level with the original,
and we claim no accuracy gain. It is not uncensored, and it is not safer either: it refuses the same kinds of
requests at the same rate.

## Which download do I need?

| You want to | Get |
|---|---|
| Run it on a GPU server with Transformers, SGLang or vLLM | this repo |
| Run it on your own computer with llama.cpp, LM Studio or Ollama | [nokeepai/Inkling-2-Mini-Prime-GGUF](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime-GGUF) |
| Apply the Prime add-on to your own copy of the original model | [nokeepai/Inkling-2-Mini-Prime-LoRA](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime-LoRA): the Prime add-on as a LoRA adapter, with the merge script |
| An uncensored model with the same Prime add-on | [nokeepai/Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) |
| The code, the per-question results and the charts | [github.com/nokeepai/inkling-2-mini-unbound](https://github.com/nokeepai/inkling-2-mini-unbound) |

A **LoRA adapter** is a small add-on file of weight changes that can be applied to a model or merged into it.

<details>
<summary><b>Where this comes from</b>: the Unbound versions, and why the Prime add-on started there</summary>

The Prime add-on was not made for Prime first. It was made for a sibling model.

[Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) is nokeep's **uncensored** Inkling 2 Mini:
the same model with its refusals removed. Each version fixed the biggest problem of the one before:

| Version | What it fixed |
|---|---|
| Unbound v1.0 | Removed the original model's refusals with a direct edit of the weights. |
| Unbound v2 and v2.1 | Made its answers genuinely useful instead of vague or evasive, by training on its own best answers. |
| Unbound v3 | Fixed over-thinking and stalling, with the Prime add-on described on this page. |

The over-thinking was not something Unbound added. The original Inkling 2 Mini has the same habit. So the same
Prime add-on was applied to the untouched original, and that is Prime. The two models share the
Prime add-on and nothing else: Prime has none of Unbound's refusal removal.

| | Refuses sensitive requests | Has the Prime add-on |
|---|---|---|
| Original Inkling 2 Mini | yes | no |
| **Prime** (this repo) | yes | yes |
| Unbound v3 | no | yes |

</details>

<details>
<summary><b>Full results</b>: public benchmarks, thinking and speed, sensitive requests, lower effort instead?</summary>

Both models were run on the same questions with the same settings (reasoning effort 0.9,
temperature 0.6, top_p 1.0, thinking limit 4096 tokens, answer
limit 8192 tokens), one pass each unless stated. Numbers in brackets are 95% confidence intervals: the range the true
value plausibly sits in, given how many items were tested.

### Did it lose anything?

Changing a model can quietly cost accuracy, so Prime was checked against the original on public
coding, maths and knowledge benchmarks. "Difference" is Prime minus the original on the same
questions. We call a result **no worse** (tables say "non-inferior") when the whole 95% interval of the difference
stays above −2 points, and **inconclusive** when the test was too small to rule out a larger loss.

| Area | Benchmarks | Questions | Original | Prime | Difference, points | Verdict |
|---|---|---|---|---|---|---|
| Coding | HumanEval + MBPP | 421 | 97.9% [96.0, 98.9] | 97.1% [95.1, 98.4] | -0.7 pts [-2.1, +0.7] | inconclusive (interval reaches past the margin) |
| Bug fixing | HumanEvalFix + QuixBugs | 195 | 96.4% [92.8, 98.2] | 94.9% [90.8, 97.2] | -1.5 pts [-4.1, +1.0] | inconclusive (interval reaches past the margin) |
| Maths | GSM8K + MATH-500 | 1,819 | 95.3% [94.2, 96.2] | 95.3% [94.2, 96.2] | +0.0 pts [-0.7, +0.7] | non-inferior |
| General knowledge | MMLU 2,000 + MMLU 400-question lab sample | 2,338 | 89.3% [88.0, 90.5] | 89.1% [87.8, 90.3] | -0.2 pts [-1.2, +0.8] | non-inferior |

![Public benchmarks, the original model against Prime, with 95% intervals](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime/resolve/main/charts/indep_prime_v3.png)

**Everything that scored lower, however slightly:**

**Below the original Inkling 2 Mini beyond noise (whole paired interval below zero):**

- **uncensoredbench delivery score:** 35.9% [34.0, 37.7] → 34.6% [32.8, 36.4], n = 1,100, paired difference -1.3 pts [-2.1, -0.4]; beyond noise.

**Lower, but inside noise:**

- **Creative writing quality:** 93.3% [78.7, 98.2] → 92.9% [77.3, 98.0], n = 30 / 28; one pass, no paired interval.
- **Roleplay quality:** 94.0% → 93.0%, n = 100; one pass, no paired interval.
- **Code problems passed first try (our own set):** 95.0% [83.5, 98.6] → 92.5% [80.1, 97.4], n = 40, paired difference -2.5 pts [-10.0, +5.0]; within noise.
- **HumanEval (independent):** 99.4% [96.6, 99.9] → 98.2% [94.8, 99.4], n = 164, paired difference -1.2 pts [-3.0, +0.0]; within noise.
- **MBPP sanitized (independent):** 96.9% [94.0, 98.4] → 96.5% [93.5, 98.2], n = 257, paired difference -0.4 pts [-2.3, +1.6]; within noise.
- **HumanEvalFix (independent):** 96.3% [92.2, 98.3] → 94.5% [89.9, 97.1], n = 164, paired difference -1.8 pts [-4.9, +0.6]; within noise.
- **MATH-500, strict format (independent):** 92.6% [90.0, 94.6] → 92.4% [89.7, 94.4], n = 500, paired difference -0.2 pts [-1.8, +1.4]; within noise.
- **MATH-500, number correct (independent):** 93.2% [90.7, 95.1] → 92.8% [90.2, 94.8], n = 500, paired difference -0.4 pts [-2.0, +1.0]; within noise.
- **MMLU (independent):** 89.7% [88.2, 90.9] → 89.5% [88.1, 90.8], n = 2,000, paired difference -0.1 pts [-1.1, +1.0]; within noise.

One coding result is unsettled: on MBPP, Prime scored -1.2 pts [-2.5, +0.1] against the original,
pooled over 3 runs with different random seeds, a possible small dip that the data neither
confirms nor rules out.

The coding and bug-fixing results are close calls, so the two largest sets were re-run three times with different
random seeds. A **seed** fixes the random choices the model makes while writing, so re-running with another seed
shows how much a score moves by chance alone:

> **Repeated over several seeds (paired, items resampled as clusters):** HumanEvalFix, 3 seeds pooled: +0.4 pts [-0.6, +1.6], no worse than the original; MBPP, 3 seeds pooled: -1.2 pts [-2.5, +0.1], inconclusive, a dip that the pooled seeds neither confirm nor rule out (on Thinking Machines' hosted service the same benchmark gave +1.6 pts [+0.4, +3.1]).
>
> **Single run, not shown to be within the margin (a loss is not ruled out):** HumanEval (-1.2 pts [-3.0, +0.0], n = 164); QuixBugs (+0.0 pts [-9.7, +9.7], n = 31).


![Checks where Prime must not be worse than the original](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime/resolve/main/charts/nothing_lost_prime_v3.png)

#### Every benchmark, one row each, plus the seed re-runs

Scores carry 95% intervals; differences are computed on the same questions for both models (paired bootstrap,
10,000 resamples). "Strict format" maths rows reject a correct number written with units, such as "230 miles",
which the original does more often; that is most of the gap between the strict and "number correct" rows. Read the
"number correct" rows. Neither reading is claimed as a gain.

| Benchmark | Source | Questions | Original | Prime | Difference, points | Verdict |
|---|---|---|---|---|---|---|
| HumanEval | openai/human-eval, all 164 problems | 164 | 99.4% [96.6, 99.9] | 98.2% [94.8, 99.4] | -1.2 pts [-3.0, +0.0] | inconclusive (interval reaches past the margin) |
| MBPP | MBPP sanitized, test split, 257 problems | 257 | 96.9% [94.0, 98.4] | 96.5% [93.5, 98.2] | -0.4 pts [-2.3, +1.6] | inconclusive (interval reaches past the margin) |
| HumanEvalFix | HumanEvalFix (HumanEvalPack, Python), 164 problems | 164 | 96.3% [92.2, 98.3] | 94.5% [89.9, 97.1] | -1.8 pts [-4.9, +0.6] | inconclusive (interval reaches past the margin) |
| QuixBugs | QuixBugs, 31 Python programs with JSON test cases | 31 | 96.8% [83.8, 99.4] | 96.8% [83.8, 99.4] | +0.0 pts [-9.7, +9.7] | inconclusive (interval reaches past the margin) |
| GSM8K, number correct | GSM8K test, all 1,319 problems | 1,319 | 96.3% [95.1, 97.2] | 96.4% [95.2, 97.2] | +0.1 pts [-0.7, +0.8] | non-inferior |
| GSM8K, strict format | GSM8K test, all 1,319 problems | 1,319 | 67.1% [64.5, 69.6] | 88.1% [86.2, 89.7] | +21.0 pts [+18.6, +23.4] | non-inferior (interval above zero; no gain claimed) |
| MATH-500, number correct | MATH-500, 500 problems | 500 | 93.2% [90.7, 95.1] | 92.8% [90.2, 94.8] | -0.4 pts [-2.0, +1.0] | non-inferior |
| MATH-500, strict format | MATH-500, 500 problems | 500 | 92.6% [90.0, 94.6] | 92.4% [89.7, 94.4] | -0.2 pts [-1.8, +1.4] | non-inferior |
| MMLU | MMLU test, 2,000 questions stratified by subject | 2,000 | 89.7% [88.2, 90.9] | 89.5% [88.1, 90.8] | -0.1 pts [-1.1, +1.0] | non-inferior |

Maths pooled with the strict format, for completeness:
GSM8K + MATH-500, 1,819 questions,
74.1% [72.0, 76.1] → 89.3% [87.8, 90.6], difference
+15.2 pts [+13.4, +17.0] (non-inferior (interval above zero; no gain claimed)).

**Seeds pooled** (every item run once per seed; paired difference with a bootstrap that resamples items as clusters, 10,000 resamples):

| Benchmark | Items × seeds | Original | Prime | Difference, points | Read-out |
|---|---|---|---|---|---|
| HumanEvalFix | 164 × 3 | 470/492 (95.5%) | 472/492 (95.9%) | +0.4 pts [-0.6, +1.6] | non-inferior |
| MBPP | 257 × 3 | 748/771 (97.0%) | 739/771 (95.8%) | -1.2 pts [-2.5, +0.1] | inconclusive (interval reaches past the margin) |


Scores per seed. Seed 1 is the run in the tables above. The table also includes the two most recent Unbound
versions (v2.1 and v3, described under "Where this comes from" above), run the same way. Spread is the highest minus
the lowest score across seeds:

| Benchmark | Model | Setup | n | Seed 1 | Seed 2 | Seed 3 | Spread, points |
|---|---|---|---|---|---|---|---|
| HumanEvalFix | Prime | our GPU server, released weights | 164 | 94.5% | 97.0% | 96.3% | 2.5 |
| HumanEvalFix | Original | our GPU server, released weights | 164 | 96.3% | 95.7% | 94.5% | 1.8 |
| MBPP | Original | our GPU server, released weights | 257 | 96.9% | 97.7% | 96.5% | 1.2 |
| MBPP | Prime | our GPU server, released weights | 257 | 96.5% | 94.9% | 96.1% | 1.6 |
| HumanEvalFix | Unbound v3 | our GPU server, released weights | 164 | 96.3% | 93.3% | 95.1% | 3.0 |
| HumanEvalFix | Unbound v2.1 | our GPU server, released weights | 164 | 95.7% | 97.6% | 95.7% | 1.9 |
| MBPP | Unbound v2.1 | our GPU server, released weights | 257 | 97.7% | 95.3% | 96.5% | 2.4 |
| MBPP | Unbound v3 | our GPU server, released weights | 257 | 96.1% | 96.5% | 96.5% | 0.4 |
| HumanEvalFix | Original | Thinking Machines' hosted service, adapter not merged | 164 | 98.2% | 96.3% |  | 1.9 |

Our own smaller test sets, one pass per model:

| Check | Items | Original | **Prime** | Difference, points |
|---|---|---|---|---|
| Everyday answers rated good by an independent grader | 63 / 62 | 96.8% [89.1, 99.1] | **96.8% [89.0, 99.1]** | |
| Creative writing rated good | 30 / 28 | 93.3% [78.7, 98.2] | **92.9% [77.3, 98.0]** | |
| Roleplay turns rated good | 100 turns | 94.0% | **93.0%** | |
| Median answer length, tokens | 100 | 284 | **293** | |
| Code problems passed first try | 40 | 95.0% [83.5, 98.6] | **92.5% [80.1, 97.4]** | -2.5 pts [-10.0, +5.0] |
| Hard maths, number correct | 100 | 94.0% [87.5, 97.2] | **97.0% [91.5, 99.0]** | +3.0 pts [-1.0, +8.0] |
| Hard maths, strict format | 100 | 73.0% [63.6, 80.7] | **91.0% [83.8, 95.2]** | +18.0 pts [+10.0, +26.0] |

The sensitive-request rows are under "Sensitive requests" below.

### Thinking and speed

Prime answers faster because it writes fewer tokens, not because the software got faster. On
everyday prompts the total per answer (thinking plus answer) fell from 1,396 to
761 tokens, and the time per answer from 119.9 s to
49.0 s, on one RTX 5090 (32 GB) (NVFP4 checkpoint as released, SGLang, 31 concurrent requests). NVFP4 is the 4-bit format
the model is released in.

![Thinking tokens and seconds per answer, the original model against Prime](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime/resolve/main/charts/speed_v3.png)

Thinking tokens per answer on 100 everyday prompts. "p90" means nine answers in ten used fewer tokens
than this; "never finished thinking" means the answer hit the thinking limit or looped:

| Model | Mean | Median | p90 | p99 | Never finished thinking |
|---|---|---|---|---|---|
| Original | 810 | 318 | 2,576 | 4,096 | 7.0% |
| **Prime** | **281** | **174** | **552** | **1,340** | **0.0%** |

### Sensitive requests

These are the 1,100 questions of uncensoredbench, a benchmark of requests that models usually refuse,
graded by the benchmark's own judge.

Prime answers as many sensitive requests as the original (64.5% [61.6, 67.2] → 65.3% [62.4, 68.0]); it
is not more uncensored. When it does answer, it answers slightly less fully (delivery score, in the table below).

| Check | Items | Original | **Prime** | Paired change, points |
|---|---|---|---|---|
| Willing to answer | 1,100 | 64.5% [61.6, 67.2] | **65.3% [62.4, 68.0]** | |
| Refused | 1,100 | 35.5% [32.8, 38.4] | **34.7% [32.0, 37.6]** | -0.8 pts [-2.3, +0.7] |
| Delivery score | 1,100 | 35.9% [34.0, 37.7] | **34.6% [32.8, 36.4]** | -1.3 pts [-2.1, -0.4] |
| Answers cut off by the thinking limit | 1,100 | 13.5% | **0.5%** | |

These rows cover all 1,100 questions (1,100 graded). No benchmark question is quoted on this page.

### Can I just turn `reasoning_effort` down instead?

**For single questions, yes: lowering the effort works about as well as the Prime add-on.** For multi-step work and
long system prompts, no. This is the most important practical question on this page, so here is the full evidence.

**Single questions.** The original at effort 0.3, and the original with the Prime add-on at
0.9, were each compared with the original at 0.9 on seven public benchmarks
(4,435 questions). Measured on Thinking Machines' hosted service with the Prime add-on applied but
not merged. Both routes cut thinking sharply, and their accuracy differences are small and similar:

| | Original, effort 0.3 | Original + Prime add-on, effort 0.9 |
|---|---|---|
| Coding (HumanEval + MBPP, 421 questions), difference in points | +1.2 pts [-0.2, +2.9] | +1.2 pts [+0.0, +2.4] |
| Bug fixing (HumanEvalFix + QuixBugs, 195) | -2.1 pts [-4.6, +0.0] | -1.5 pts [-4.1, +0.5] |
| Maths, number correct (GSM8K + MATH-500, 1,819) | +0.1 pts [-0.5, +0.7] | +0.2 pts [-0.4, +0.9] |
| Knowledge (MMLU, 2,000) | -0.2 pts [-1.3, +0.8] | +0.8 pts [-0.2, +1.8] |
| Mean thinking tokens per question (original at 0.9: 1,010) | 390 | 462 |
| Answers cut off by the thinking limit (original at 0.9: 10.2% [9.4, 11.1]) | 1.0% [0.8, 1.4] | 1.3% [1.0, 1.7] |

**Multi-step debugging.** Here lowering the effort helps but does not cure the stalling. The same
15 hard debugging tasks at three efforts, on the same hosted service:

| | Effort 0.3 | Effort 0.6 | Effort 0.9 |
|---|---|---|---|
| Original | 100% solved, 19% of turns with no code, 4,553 tokens per task | 100% solved, 52% of turns with no code, 8,090 tokens per task | 80% solved, 88% of turns with no code, 26,784 tokens per task |
| Original + Prime add-on | 100% solved, 0% of turns with no code, 1,825 tokens per task | 100% solved, 0% of turns with no code, 2,478 tokens per task | 100% solved, 0% of turns with no code, 2,491 tokens per task |

With the Prime add-on, the result is the same at every effort, so there is nothing to tune. This repo's weights at
effort 0.3 on a GPU server: 100% solved, 0% of turns with no code.

**Long system prompts.** Every rule followed: the original 112/160 (70.0%) at effort 0.9 and
25/40 (62.5%) at 0.3; with the Prime add-on 148/160 (92.5%) at 0.9 and
35/40 (87.5%) at 0.3. Lowering the effort does not help the original here. Measured on Thinking
Machines' hosted service with the Prime add-on applied but not merged into the weights.

#### The same comparison on this repo's weights, on a GPU server

The release setup (SGLang, 31 concurrent requests, NVFP4 checkpoint as released), each model at both efforts. Hard debugging is
15 tasks; the benchmarks are the full sets.

| Model and effort | Hard debugging solved | Turns with no code | Tokens per task | HumanEvalFix | MBPP | MATH-500 | GSM8K, number correct | MMLU |
|---|---|---|---|---|---|---|---|---|
| **Prime**, effort 0.9 | 100.0% [79.6, 100.0] | 0.0% | 2,766 | 94.5% [89.9, 97.1] | 96.5% [93.5, 98.2] | 92.4% [89.7, 94.4] | 96.4% [95.2, 97.2] | 89.5% [88.1, 90.8] |
| Prime, effort 0.3 | 100.0% [79.6, 100.0] | 0.0% | 2,182 | 95.7% [91.5, 97.9] | 96.9% [94.0, 98.4] | 93.2% [90.7, 95.1] | 96.9% [95.8, 97.7] | 89.3% [87.9, 90.6] |
| Original, effort 0.9 | 80.0% [54.8, 93.0] | 86.2% | 23,920 | 96.3% [92.2, 98.3] | 96.9% [94.0, 98.4] | 92.6% [90.0, 94.6] | 96.3% [95.1, 97.2] | 89.7% [88.2, 90.9] |
| Original, effort 0.3 | 100.0% [79.6, 100.0] | 25.0% | 5,038 | 97.0% [93.1, 98.7] | 96.1% [93.0, 97.9] | 93.2% [90.7, 95.1] | 96.5% [95.4, 97.4] | 89.6% [88.2, 90.9] |

Strict-format maths was left out of the hosted-service table because it measures answer formatting: strict
difference +6.4 pts [+5.0, +7.9] at low effort, +14.5 pts [+12.8, +16.3] with the Prime
add-on.

**The honest conclusion.** If you only ask single questions, the original model at a lower effort is a fine
choice. Prime is worth it when the model works in several steps (debugging loops, agents, tool use)
or follows a long system prompt, and when you would rather not tune the effort at all.

### Against Unbound and other releases

This table puts Prime next to the original model, every Unbound version (see "Where this comes from"
above) and some uncensored Inkling 2 Mini releases by others. Each column says which release measured it. Blank
means that release did not measure it; nothing is estimated. "Useful" columns measure answers to harmful requests,
which the original and Prime refuse; Prime was not run on that test. **Drift** is how
far a model's word-by-word predictions on ordinary text have moved from the original's (KL divergence; smaller means
closer to the original).

| Model | Useful, thinking off (v2.1 release) | Useful, thinking on (v2.1 release) | Refused, off / on (v2.1 release) | MMLU-300 / GSM8K-100 (v2.1 release) | Drift from the original (v3 release) | Sensitive requests: willing (v3 release) | Hard debugging solved (v3 release) | MMLU, 2,000 questions (v3 release) |
|---|---|---|---|---|---|---|---|---|
| Original Inkling 2 Mini | 0.8% [0.0, 1.9] | 0.8% [0.0, 1.9] | 96.2% / 95.8% | 76.3 / 97 | reference (0 by definition) | 64.5% [61.6, 67.2] | 80.0% [54.8, 93.0] | 89.7% [88.2, 90.9] |
| Inkling-2-Mini-Prime |  |  |  |  | 0.004928 ± 0.000146 (same top token 96.84%) | 65.3% [62.4, 68.0] | 100.0% [79.6, 100.0] | 89.5% [88.1, 90.8] |
| Unbound v1.0 | 51.3% [45.2, 57.4] | 61.6% [55.9, 67.3] | 0.0% / 0.0% | 75.7 / 99 | 0.011570 ± 0.000304 (same top token 95.39%) |  |  |  |
| Unbound v2 | 77.9% [73.0, 82.9] | 82.5% [77.9, 86.7] | 0.4% / 1.1% | 76.0 / 99 | 0.037977 ± 0.000710 (same top token 94.549%) |  |  |  |
| Unbound v2.1 | 83.7% [79.1, 87.8] | 85.2% [80.6, 89.4] | 0.4% / 0.4% | 74.3 / 98 | 0.048836 ± 0.000479 (same top token 94.324%) | 99.4% [98.7, 99.7] | 73.3% [48.0, 89.1] | 87.7% [86.1, 89.0] |
| **Unbound v3** |  |  |  |  | 0.051909 ± 0.000505 (same top token 94.167%) | 99.9% [99.5, 100.0] | 100.0% [79.6, 100.0] | 87.2% [85.6, 88.5] |
| IndrasMirror XQ | 58.9% [53.2, 64.6] | 68.1% [62.4, 73.8] | 0.0% / 0.0% | 74.7 / 98 |  |  |  |  |
| AtomicChat Uncensored, LoRA | 52.1% [46.0, 58.2] | 55.5% [49.4, 61.6] | 0.0% / 0.0% | 78.3 / 99 |  |  |  |  |
| AtomicChat Uncensored, AD-Q4_K file | 50.2% [44.1, 56.3] | 59.3% [53.2, 65.4] | 0.4% / 0.0% | 77.0 / 98 |  |  |  |  |
| OpenUncensored | 33.1% [27.4, 38.8] | 41.8% [35.7, 47.9] | 1.5% / 0.8% | 75.3 / 98 |  |  |  |  |
| JoaoZaokk Heretic | 17.5% [12.9, 22.4] | 26.2% [20.9, 31.6] | 1.1% / 0.4% |  |  |  |  |  |

![Prime, the original model, the Unbound versions and other releases side by side](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime/resolve/main/charts/xver_v3.png)

</details>

<details>
<summary><b>Do the smaller GGUF files keep the add-on's effect?</b>: mostly yes; hard debugging re-run on Q8_0, Q4_K_M and IQ2_M</summary>

Mostly yes: the smallest file, IQ2_M, still solves 100.0% of the tasks but stalls on
5.6% of turns; see the table. GGUF files are compressed ("quantised") copies of the weights;
Q8_0 is the largest and closest to the original weights, IQ2_M the smallest. Hard debugging re-run on the files with
llama.cpp (15 tasks, same task set):

| File | Hard debugging solved | Turns with no code | Tokens per task | GSM8K, first 100, number correct | HumanEval, first 50 |
|---|---|---|---|---|---|
| Prime Q8_0 | 100.0% | 0.0% | 2,606 | 94.0% | 100.0% |
| **Prime Q4_K_M** | **100.0%** | **0.0%** | **2,367** | **98.0%** | **98.0%** |
| Prime IQ2_M | 100.0% | 5.6% | 3,261 | 95.0% | 100.0% |
| Original Q4_K_M, made the same way | 80.0% | 87.9% | 26,939 | 97.0% | 100.0% |

How the files were built and checked is on the [GGUF card](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime-GGUF).

</details>

<details>
<summary><b>Files in this repo</b>: the weights, configs and charts</summary>

| Path | What it is |
|---|---|
| `model-00001-of-00001.safetensors`, `mtp.safetensors`, configs, tokenizer, chat template | Prime, ready to load. The sha256 of each file is in this repo's file list. |
| `charts/` | The charts on this page |

</details>

<details>
<summary><b>Limits</b>: where it can fall short, and what was not measured</summary>

- **No accuracy gain, and small coding losses not ruled out.** See "Did it lose anything?" under "Full results"
  above.

- **When it does hit the thinking limit, the answer is usually wrong.** Prime rarely runs into the
  limit, but when it does, the answer it is then forced to give tends to fail. The original's forced answers are
  usually still right. HumanEvalFix, one pass (after the 4,096-token cap the model is made to answer), 164 questions: the original hit the limit
  32 times and 26 of those answers passed; Prime hit it
  7 times and 1 passed. On Thinking Machines' hosted service: the original
  30 cap hits, 27 passed (second seed: 28, 22); with the Prime add-on 7 cap hits, 2 passed (second seed: 10, 4). The Prime add-on makes hitting the limit
  rare; it does not make a cut-off answer good.

- **Shorter thinking can miss things.** Problems that genuinely need long reasoning may suffer.
- **Very close to the original, not identical.** Drift (how far its word-by-word predictions on ordinary text have
  moved from the original's): 0.004928 ± 0.000146 (same top token 96.84%). Expect some wording differences.
- **Small debugging test.** The headline debugging result is 15 tasks. The direction
  is clear; the exact percentages are not precise.
- **One pass per model.** Apart from the seed re-runs, every number is a single run. Differences of one or two
  points on the small coding sets can be chance.
- **Not measured:** long documents, images and tool calling. Only English was tested.
- Everything the [original model's card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) says about capabilities,
  safety and limits still applies.

</details>

<details>
<summary><b>How it was made</b>: training method, data, checkpoint choice and merge</summary>

- **Starting point.** The original [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), untouched.
- **The Prime add-on.** A LoRA adapter trained with **reinforcement learning**: the model attempted tasks and was
  rewarded for finishing them correctly without wasted thinking. Recipe: LoRA rank 32 on the attention and expert layers, 60 steps, 32 prompts × 8 samples per step, learning rate 5e-5, thinking turned off on 25% of samples, reasoning effort 0.5 to 0.9, trained from the original model on harmless prompts only; 619 anchor answers (the original model's own best-of-8 answers, answer part only) hold the answer style. In plain terms: the
  learning rate sets how big each training update is; anchor answers are the original model's own answers (for each
  prompt, the best of eight attempts, "best-of-8"), trained on alongside so the model keeps its answer style.
- **Which checkpoint shipped.** We shipped step 60 of the final training run. The reason: a 25-step continuation with extra roleplay, system-prompt and long-answer data measured level with step 60 on every check, so step 60 shipped.
- **Training data.** 1,185 harmless prompts written for this project (by step: debugging 31%, maths 19%, creative writing 16%, questions 16%, roleplay 9%, chat 9%). No
  harmful requests and no benchmark questions. To keep answer quality from drifting, each step also trained on
  619 of the original model's own answers.
- **Was it trained on the test questions?** No. To check, every item's prompt (first 120 normalised characters) was searched for in all training prompts and anchor answers: 0 hits across all nine benchmark files (4,435 unique independent items plus the two lab samples). One overlap to disclose: 35 MBPP, 50 GSM8K
  and 5 MATH-500 questions also appear in the small in-house test sets used to pick the final checkpoint, so they
  were seen when choosing between checkpoints, never in training.
- **Merge.** The adapter was merged into the original weights (stochastic (dithered) rounding into the quantised weights, attention and experts). The merge is repeatable: it
  produced a byte-identical file in 3 separate runs (sha256 `823f44bbcbdcb50e790cde840b310c71acca4c2317c9ef806e47fc29f81c827b`).

</details>

<details>
<summary><b>Reproduce the evaluation</b>: commands, settings, rebuilding the weights, per-question results</summary>

Evaluation code is in `scripts/evalsuite/` in the [GitHub repo](https://github.com/nokeepai/inkling-2-mini-unbound). Served
with SGLang (image lmsysorg/sglang:inkling-2-mini, pinned by digest in scripts/v3stack/stack_test.sh), NVFP4 checkpoint as released, 31 concurrent requests on one RTX 5090 (32 GB). Settings: reasoning effort 0.9, temperature 0.6, top_p 1.0, thinking cap 4,096 tokens, answer cap 8,192 tokens, seed 1234, one pass per model. The suite names and `--tag` values
in the commands are the harness's own labels; `bison` and `haiku` are the two grader models (Bison and Claude Haiku).

```bash
# generate (one tag per model; all uncensoredbench questions (no --limit))
python3 scripts/evalsuite/run_suite.py --base-url http://127.0.0.1:8000/v1 --model MODEL --tag prime \
    --out-root results --suites agent_hard_noforce,ubench,main,hardmath,code,rp,pub_all \
    --effort 0.9 --temperature 0.6 --top-p 1.0 \
    --think-cap 4096 --answer-cap 8192 --seed 1234 --repeats 1 --allow-effort-unverified
# grade (paid judges): ubench with the grader model Bison; main, hardmath and rp with the two grader models (Bison and Claude Haiku)
python3 scripts/evalsuite/run_suite.py --base-url http://unused/v1 --model x --tag prime --out-root results \
    --suites ubench --judge-only --judges bison --ubench-conditions command
python3 scripts/evalsuite/run_suite.py --base-url http://unused/v1 --model x --tag prime --out-root results \
    --suites main,hardmath,rp --judge-only --judges bison,haiku
# intervals and paired tests, the original ("stock") as the reference
python3 scripts/evalsuite/compare.py stock prime --ci --ref stock --margin 2 --out-root results --json ci.json \
    --suites pub_humaneval,pub_mbpp,pub_humanevalfix,pub_quixbugs,pub_gsm8k_full,pub_math500,pub_mmlu_2k,code,hardmath
```

To rebuild Prime from the original model and the adapter: `merge_adapter.py --parts attn_experts
--rounding stochastic --seed 0 --device cuda` (about 2.5 minutes on one RTX 5090; script and adapter in
[nokeepai/Inkling-2-Mini-Prime-LoRA](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime-LoRA); adapter sha256 `a13961f470a8b804b28f1c065c6b770ba6a0c8f449d554292d3f6ea9f8b6fdb9`).
The same adapter merged into Unbound v2.1 gives Unbound v3 (sha256 `88e78b0126722a99dcecf26dd7b6cf242860d415f4f74b651e57b18d6abf3b88`, byte-identical in
2 runs).


Per-question scores for every benchmark (no model text for the sensitive-request benchmark):
https://github.com/nokeepai/inkling-2-mini-unbound/tree/main/package_v3/results/data


</details>

## Intended use

For anyone who wants the original Inkling 2 Mini's behaviour, including its refusals, with less waiting and fewer
stalls: coding assistants, agents that work in several steps, chat with long system prompts, and everyday
questions. It is not an uncensored model; if that is what you need, see
[Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound). The base model's
[Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) applies.

## Licence

Apache 2.0, like the base model, Inkling 2 Mini by Thinking Machines Lab, Inc.

## About nokeep

Built and paid for by [nokeep](https://nokeep.ai/?ref=huggingface): private, uncensored AI chat, images and video.

## Citation

```bibtex
@misc{prime_2026,
  title  = {Inkling-2-Mini-Prime: Inkling 2 Mini that finishes instead of over-thinking},
  author = {nokeep},
  year   = {2026},
  url    = {https://huggingface.co/nokeepai/Inkling-2-Mini-Prime}
}
```
