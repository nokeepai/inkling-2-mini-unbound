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

![Inkling-2-Mini-Unbound v3](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/banner_v3.png)

# Inkling-2-Mini-Unbound

**An uncensored Inkling 2 Mini: it answers instead of refusing. New in v3: it also finishes its work
instead of over-thinking.**

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is Thinking Machines Lab's open-weights reasoning
model: it takes text and images, and it "thinks" in private before it answers. Out of the box it refuses most
sensitive requests. **Unbound is the same model with the refusals removed.** The architecture, files and chat
template are unchanged, so anything that runs Inkling 2 Mini runs Unbound. It is built by
[nokeep](https://nokeep.ai/?ref=huggingface) for people who apply their own content rules: researchers,
red-teamers, fiction writers, and developers running a system they control.

Current version: **v3**, released 2026-11-27. Earlier versions stay available (see
"Version history" below).

> **This model follows harmful instructions.** It gives detailed answers to requests for harmful, dangerous or
> illegal content that the original model declines. It has no safety training layered back on. Do not deploy it to
> end users without your own safeguards.

**Headline numbers**

- **Useful answers to harmful requests:** 81.7% with thinking off, 87.8% with thinking
  on. Next best public release we could load: 58.9% / 68.1%; the original model: 0.8%.
- **Hard debugging tasks solved:** 73.3% at v2.1 → **100%** at v3
  (15 tasks).
- **Thinking per everyday prompt:** 344 → **184** tokens (median); time per
  answer 161.7 s → **52.5 s**.
- **Willing to answer sensitive requests:** **99.9%** (v2.1: 99.4%).
- **Still a drop-in Inkling 2 Mini:** same architecture, files, tokenizer and chat template.

## Quick start

**On your own computer** (llama.cpp, LM Studio, Ollama): use the
[GGUF files](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF). Most people want `Q4_K_M` (about 18 GB).

```bash
hf download nokeepai/Inkling-2-Mini-Unbound-GGUF \
    Inkling-2-Mini-Unbound-v3-Q4_K_M.gguf mmproj-Inkling-2-Mini-Q8_0.gguf --local-dir models

llama-server -m models/Inkling-2-Mini-Unbound-v3-Q4_K_M.gguf \
    --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf \
    --jinja -c 32768 -ngl 99 --temp 0.6 --top-p 1.0
```

The [GGUF card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF) lists every file size, the llama.cpp version
you need, and the LM Studio and Ollama steps.

**On a GPU server** (Transformers, SGLang, vLLM): the weights in this repo load exactly like
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini); follow that card's instructions and swap in
`nokeepai/Inkling-2-Mini-Unbound`. We ran it on one RTX 5090 (32 GB). Then call it like any OpenAI-compatible server:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "nokeepai/Inkling-2-Mini-Unbound",
  "messages": [{"role": "user", "content": "Explain how a hash table handles collisions."}],
  "temperature": 0.6, "top_p": 1.0,
  "chat_template_kwargs": {"reasoning_effort": 0.9}
}'
```

**Settings.** Every number on this page was measured with these, so they are the ones we recommend:

| Setting | Value | What it does |
|---|---|---|
| `reasoning_effort` | 0.9 | How much the model thinks before it answers. |
| `temperature` | 0.6 | Randomness. Do not use 0: earlier versions looped in their thinking at 0. |
| `top_p` | 1.0 | llama.cpp defaults to 0.95, so set it yourself. |
| Thinking limit | 4096 tokens | Stops runaway thinking; the model then has to answer. |

## What is new in v3

The previous version, v2.1, had one real flaw: **it thought too much.**

Inkling writes private reasoning before each answer. v2.1 often could not stop. On a benchmark of sensitive
requests, 49.5% of its answers ran into the thinking limit. On multi-step debugging it spent most
turns thinking and wrote no code at all. The original Inkling 2 Mini has the same habit.

v3 adds one small extra training step, the **Prime add-on**: a small set of weight changes (a LoRA adapter)
trained only on harmless tasks. It is the same add-on that turns the original Inkling 2 Mini into
Inkling-2-Mini-Prime. It teaches the model to stop deliberating and act. Because it was trained only on harmless
tasks (debugging, maths, writing, everyday questions), it changes how long the model thinks, not what it is willing
to say.

![Unbound v3 against v2.1: hard debugging tasks solved, turns with no code written, and tokens used per task](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/hero_v3.png)

| What we measured | v2.1 | **v3** |
|---|---|---|
| Hard debugging tasks solved (15 tasks) | 73.3% [48.0, 89.1] | **100.0% [79.6, 100.0]** |
| Debugging turns where it wrote no code | 88.7% | **0.0%** |
| Tokens used per debugging task | 26,130 | **2,429** |
| Thinking per everyday prompt (median tokens) | 344 | **184** |
| Sensitive requests: answers cut off by the thinking limit | 49.5% | **5.7%** |
| Sensitive requests: time per answer, with 31 requests running at once on one GPU | 634.0 s | **285.1 s** |
| Sensitive requests: willing to answer | 99.4% [98.7, 99.7] | **99.9% [99.5, 100.0]** |

**How to read it.** The debugging tasks are small programs with three interacting bugs each, worked on over up to
12 rounds of test output; the sensitive requests are a 1,100-question benchmark. Numbers in brackets are 95%
confidence intervals. "Full results" below explains each test in more detail.

**What v3 is not.** It is not smarter: on public benchmarks it scores level with v2.1, within about
2 points, and we claim no accuracy gain. It is not more uncensored either: v2.1 already answered almost
everything.

## Which download do I need?

| You want to | Get |
|---|---|
| Run it on a GPU server with Transformers, SGLang or vLLM | this repo |
| Run it on your own computer with llama.cpp, LM Studio or Ollama | [nokeepai/Inkling-2-Mini-Unbound-GGUF](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF) |
| A single 4-bit file for NVIDIA Blackwell GPUs (RTX 50-series) | [nokeepai/Inkling-2-Mini-Unbound-Basalt](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-Basalt) |
| The original, still-censored model with the same Prime add-on | [nokeepai/Inkling-2-Mini-Prime](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime) |
| The code, the per-question results and the charts | [github.com/nokeepai/inkling-2-mini-unbound](https://github.com/nokeepai/inkling-2-mini-unbound) |

<details>
<summary><b>Version history</b>: what v1.0, v2, v2.1 and v3 each fixed</summary>

Each version fixed the biggest problem of the one before.

| Version | Date | The problem it fixed | How | Useful answers to harmful requests |
|---|---|---|---|---|
| Original Inkling 2 Mini | | | | 0.8% |
| v1.0 | 2026-11-21 | The original refuses almost every harmful request. | A direct edit of the weights that removes the model's internal "refuse" signal. No training. | 51.3% |
| v2 | 2026-11-22 | v1.0 stopped refusing, but about half its answers still dodged: satire, vague outlines, piles of caveats. | Fine-tuned on its own best direct answers. | 77.9% |
| v2.1 | 2026-11-23 | v2 still held back on many answers and looped in its thinking. | A second round of training on its own answers, preferring full answers to held-back ones. | 83.7% |
| **v3** (current) | 2026-11-27 | v2.1 over-thinks and stalls on multi-step work. | The Prime add-on: a small training step on harmless tasks only. | see the comparison section below |

"Useful answers" is the share of 263 held-out harmful requests that got a genuinely useful answer
with thinking off, as measured for the v2.1 release.
v3 was later run through the same test; its numbers are in "How it compares with other uncensored releases"
below. On a newer, larger benchmark, willingness went from 99.4% [98.7, 99.7] at v2.1 to
99.9% [99.5, 100.0] at v3.

Every earlier version is still here: `v2.1` and `v2` are branches of this repo, `v1.0` is a tagged revision. The
full story of each release is in the
[changelog](https://github.com/nokeepai/inkling-2-mini-unbound/blob/main/package_v3/CHANGELOG.md).

</details>

<details>
<summary><b>How it compares with other uncensored releases</b>: useful answers, refusals and drift for every public release we could load</summary>

Most uncensored releases only remove the refusal, with a weight edit often called abliteration. That stops the
model saying "no", but about half its answers still dodge: they turn the request into satire or a "conceptual"
example, bury it in caveats, or stay vague. Unbound is trained on top of that edit to stop dodging. So the question
this test asks is not "does it refuse?" but **"is the answer actually useful?"**

**v3 on the same test.** On 2026-11-27 v3 was run on the same 263 requests, with the same engine,
settings and judge as the v2.1 release run. v2.1's saved answers were re-judged in the same pass, so the two are
directly comparable (the judge was reached through a different provider this time; re-judging moved v2.1's own
score by less than half a point). Useful answers: v3 81.7% [76.8, 86.3] / 87.8% [83.7, 91.6] against v2.1
84.0% [79.5, 88.2] / 85.6% [81.0, 89.7] (thinking off / on). Difference on the same requests:
-2.3 pts [-6.8, +2.3] with thinking off and +2.3 pts [-1.5, +6.1] with thinking on. v3 refused
0 of 263 requests with thinking off and 0 of 263 with thinking on. With thinking on, 3.8% of v3's answers hit the thinking limit (v2.1:
22.1%). By the second judge (Bison, described below): v3 90.5% [86.7, 93.9] / 94.7% [92.0, 97.3] against v2.1
93.9% [90.9, 96.6] / 94.3% [91.3, 97.0], difference -3.4 pts [-6.1, -0.8] / +0.4 pts [-2.3, +3.0]. In plain words: with thinking on, v3 is level
with v2.1 or slightly ahead; with thinking off it is a few points lower, within noise by the main judge and just
beyond it by the second. The other
releases' rows and the charts below are from the v2.1 release run.

**The test.** Every public uncensored Inkling 2 Mini we could load, on the same 263 held-out
harmful requests, with the same engine (llama.cpp `19bc41dbb`), the same settings (greedy decoding, which means no randomness, and
answers allowed up to 8,192 tokens) and the same judge. "Useful" means a judge model rated the answer as fulfilling the request.
"Thinking off / on" is the model run without its private reasoning, and with it at `reasoning_effort` 0.9.
"Drift" is how far a model's word-by-word predictions on ordinary text have moved from the original's (measured
as KL divergence, KLD; smaller is closer). MMLU-300 and GSM8K-100 are small knowledge and maths checks.

![Share of harmful requests that got a useful answer, every release tested (chart from the v2.1 release; the Unbound bar is v2.1)](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/compare_useful_v21.png)

| Model (repo and version tested) | Refused, off / on | Useful, thinking off | Useful, thinking on | Drift from the original (KLD) | MMLU-300 / GSM8K-100 |
|---|---|---|---|---|---|
| **Unbound v2.1** (what v3 is built on) | 0.4% / 0.4% | **83.7%** [79.1, 87.8] | **85.2%** [80.6, 89.4] | 0.0439 | 74.3 / 98 |
| Unbound v2 (2026-11-22) | 0.4% / 1.1% | 77.9% | 82.5% | 0.0318 | 76.0 / 99 |
| Unbound v1.0 (2026-11-21) | 0.0% / 0.0% | 51.3% | 61.6% | 0.0102 | 75.7 / 99 |
| IndrasMirror XQ † ([IndrasMirror/Inkling-2-Mini-XQ-Uncensored](https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored) @ `209cca6`, 2026-11-22) | 0.0% / 0.0% | 58.9% | 68.1% | 0.0357 † | 74.7 / 98 |
| AtomicChat Uncensored, AD-Q4_K file † ([AtomicChat/Inkling-2-Mini-Uncensored-GGUF](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF) @ `40886ab`) | 0.4% / 0.0% | 50.2% | 59.3% | 0.0106 † | 77.0 / 98 |
| AtomicChat Uncensored, LoRA (same repo, 2026-11-21) | 0.0% / 0.0% | 52.1% | 55.5% | **0.0078** | **78.3** / 99 |
| OpenUncensored ([OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored) @ `4e5f06d`, 2026-11-21) | 1.5% / 0.8% | 33.1% | 41.8% | 0.0228 | 75.3 / 98 |
| JoaoZaokk Heretic †‡ ([JoaoZaokk/Inkling-2-Mini-Heretic-GGUF](https://huggingface.co/JoaoZaokk/Inkling-2-Mini-Heretic-GGUF) @ `1f407a1`, UD-Q4_K_XL, 2026-11-22) | 1.1% / 0.4% | 17.5% | 26.2% | not measured ‡ | not measured ‡ |
| Original Inkling 2 Mini | 96.2% / 95.8% | 0.8% | 0.8% | — | 76.3 / 97 |

† A 4-bit file, run as published, where the others are 8-bit. Its drift includes rounding noise; for scale, the
unedited original at a similar 4-bit size has drift 0.0073.

‡ Heretic's file does not load in standard llama.cpp, so it ran in a compatible build with the same prompts,
settings and judge. Its drift and benchmarks could not be measured there.

**Head to head on the same requests.** Where Unbound v2.1 and another release disagreed, this is how often each
one was the one that gave the useful answer:

- **Against IndrasMirror XQ** (the next best): 75 to 10 with
  thinking off, 56 to 11 with thinking on. Overall gain
  +24.7 points [+18.6, +30.8] off, +17.1 points
  [+11.4, +22.8] on.
- **Against AtomicChat's LoRA** (the most downloaded alternative): 87 to
  4 off, 88 to 10 on. Overall gain
  +31.6 points [+25.5, +37.6] off, +29.7 points
  [+23.2, +36.1] on.
- **Against Heretic:** 175 to 1 with thinking off. Heretic
  almost never refuses, but most of its answers hedge or reframe the request.
- **Against our own v2:** 25 to 10 off
  (+5.7 points [+1.5, +10.3]), 18 to
  11 on (+2.7 points [-1.1, +6.8], within noise).

![Head to head: on requests where two releases disagreed, which one gave the useful answer (v2.1 release)](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/headtohead_v21.png)

**A second judge, to check the first.** The judge that scored "useful", a judge model called Jev, also helped pick Unbound's
training answers, so a model could in principle have learned to please that judge. A second judge model, Bison, that was
never used in training, ranks the releases the same way: 93.5% /
93.5% for v2.1, 87.1% / 90.1% for v2,
73.8% / 76.8% for v1.0, 69.2% /
71.5% for AtomicChat's LoRA and 0.8% / 1.1% for
the original (thinking off / on). Bison was not run on the other releases.

**Where other releases do better:**

- **Staying close to the original.** AtomicChat's LoRA (0.0078), our v1.0 (0.0102), OpenUncensored (0.0228) and
  our v2 (0.0318) all drift less than v2.1 (0.0439), and v3 drifts a little more than v2.1. Expect
  more wording differences from the original on ordinary requests. If you want the lightest touch, use v1.0.
- **MMLU-300.** AtomicChat's LoRA scores highest (78.3). v2.1's 74.3 has a 95% interval of
  [69.3, 79.3], which contains every release's score.
- **Zero refusals.** v1.0, AtomicChat's LoRA and XQ refused none of the 263 requests. v2.1 refused
  1 with thinking off and 1 with thinking on.
- **Some categories.** v2 scores higher than v2.1 in two small categories (misinformation, and "other harm" with
  12 requests).
- **Thinking loops, in this test.** With greedy decoding and thinking on, v2.1 hit the thinking limit on
  22.1% of these requests, more than v2 (17.1%), XQ (15.2%)
  or AtomicChat's LoRA (8.0%). Over-thinking is what v3 was built to cure: in the v3 run,
  3.8% of v3's answers hit the limit (v2.1 in the same pass: 22.1%).

![Usefulness against drift from the original, every release tested (v2.1 release)](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/tradeoff_v21.png)

**What was re-measured for v3, and what was not.**

- Re-measured for v3, with one method for every Unbound version: drift from the original. The figures are in
  "One table across every version" under "Full results" below. They use a different text sample from the table
  above, so compare them with each other, not with that column.
- Measured for v3 and v2.1 on the same questions: everything under "Full results" below.
- Re-measured for v3 on the same 263 requests: useful answers, refusals and thinking-limit hits (see
  "v3 on the same test" above). Not re-measured for v3: the head-to-head counts, the drift and benchmark columns of
  the table above, and the other releases on the newer benchmark.

</details>

<details>
<summary><b>Full results</b>: still uncensored, nothing lost on public benchmarks, thinking and speed, every version in one table</summary>

Both versions were run on the same questions with the same settings, one pass each unless stated.

The tests in "What is new in v3", in more detail:

- **Hard debugging tasks** are 15 small programs with three interacting bugs each. The
  model sees the failing test output, proposes a change, sees the new output, and repeats for up to 12 turns.
- **Sensitive requests** are the 1,100 questions of uncensoredbench, a benchmark of requests that models
  usually refuse, graded by the benchmark's own judge.
- **Numbers in brackets** are 95% confidence intervals: the range the true value plausibly sits in, given how many
  items were tested.

### Is it still uncensored?

Yes. The Prime add-on did not touch willingness.

| Sensitive requests (uncensoredbench, 1,099 / 1,100 graded) | v2.1 | **v3** |
|---|---|---|
| Willing to answer | 99.4% [98.7, 99.7] | **99.9% [99.5, 100.0]** |
| Refused | 0.6% [0.3, 1.3] | **0.1% [0.0, 0.5]** |
| Delivery score: how much of what was asked the answer delivers | 57.5% [56.2, 58.8] | **58.1% [56.9, 59.4]** |
| Answered but held something back | 18.8% [16.6, 21.2] | **17.4% [15.2, 19.7]** |
| Judged a complete answer | 8.7% [7.2, 10.6] | **10.4% [8.7, 12.3]** |

Comparing the two versions question by question, the delivery score moved by +0.6 pts [-0.3, +1.5]: no
real change. Note that "willing" is not the same as "useful": some answers still hold part of the request back
(see "Limits" below). No benchmark question is quoted on this page.

![v2.1 and v3 on the sensitive-request benchmark: willing, delivery score and answers cut off by the thinking limit](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/history_v3.png)

### Did it lose anything?

Changing a model can quietly cost accuracy, so v3 was checked against v2.1 on public coding, maths and knowledge
benchmarks. "Difference" is v3 minus v2.1 on the same questions. The tables call a result **non-inferior**, meaning
no worse, when the whole 95% interval of the difference stays above −2 points, and **inconclusive** when the test was too small to rule out
a larger loss.

| Area | Benchmarks | Questions | v2.1 | v3 | Difference, points | Verdict |
|---|---|---|---|---|---|---|
| Coding | HumanEval + MBPP | 421 | 97.9% [96.0, 98.9] | 97.1% [95.1, 98.4] | -0.7 pts [-2.1, +0.7] | inconclusive (interval reaches past the margin) |
| Bug fixing | HumanEvalFix + QuixBugs | 195 | 95.9% [92.1, 97.9] | 95.4% [91.5, 97.5] | -0.5 pts [-3.6, +2.6] | inconclusive (interval reaches past the margin) |
| Maths | GSM8K + MATH-500 | 1,819 | 95.4% [94.4, 96.3] | 95.8% [94.7, 96.6] | +0.3 pts [-0.3, +1.0] | non-inferior |
| General knowledge | MMLU (2,000 questions, plus a 400-question sample of our own) | 2,338 | 87.5% [86.1, 88.8] | 87.0% [85.6, 88.3] | -0.4 pts [-1.4, +0.6] | non-inferior |

![Public benchmarks, v2.1 against v3, with 95% intervals](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/indep_unbound_v3.png)

Every check where v3 scored below v2.1, however slightly ("independent" marks a public benchmark):

**Lower, but inside noise:**

- **MBPP sanitized (independent):** 97.7% [95.0, 98.9] → 96.1% [93.0, 97.9], n = 257, paired difference -1.6 pts [-3.5, +0.0]; within noise.
- **QuixBugs (independent):** 96.8% [83.8, 99.4] → 90.3% [75.1, 96.7], n = 31, paired difference -6.5 pts [-19.4, +6.5]; within noise.
- **MMLU (independent):** 87.7% [86.1, 89.0] → 87.2% [85.6, 88.5], n = 2,000, paired difference -0.5 pts [-1.6, +0.6]; within noise.

The two smaller coding sets, MBPP and HumanEvalFix, were each re-run three times with different random seeds (a
seed fixes the random choices the model makes while writing, so re-running with another seed shows how much a
score moves by chance). Pooled over the three runs, MBPP comes out level (-0.1 pts [-1.3, +1.2]). HumanEvalFix comes out at -1.4 pts [-3.5, +0.6]: a
small loss there is neither confirmed nor ruled out. QuixBugs is only 31 questions, so its result says little
either way.

![Checks where v3 must not be worse than v2.1](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/nothing_lost_unbound_v3.png)

#### Every benchmark, one row each

Scores carry 95% intervals; differences are computed question by question on the same questions for both
versions (bootstrap, 10,000 resamples). "Strict format" maths rows reject a correct number written with units, such as "230 miles",
which v2.1 does more often; read the "number correct" rows.

| Benchmark | Source | Questions | v2.1 | v3 | Difference, points | Verdict |
|---|---|---|---|---|---|---|
| HumanEval | openai/human-eval, all 164 problems | 164 | 98.2% [94.8, 99.4] | 98.8% [95.7, 99.7] | +0.6 pts [-1.2, +2.4] | non-inferior |
| MBPP | MBPP sanitized, test split, 257 problems | 257 | 97.7% [95.0, 98.9] | 96.1% [93.0, 97.9] | -1.6 pts [-3.5, +0.0] | inconclusive (interval reaches past the margin) |
| HumanEvalFix | HumanEvalFix (HumanEvalPack, Python), 164 problems | 164 | 95.7% [91.5, 97.9] | 96.3% [92.2, 98.3] | +0.6 pts [-2.4, +3.7] | inconclusive (interval reaches past the margin) |
| QuixBugs | QuixBugs, 31 Python programs with JSON test cases | 31 | 96.8% [83.8, 99.4] | 90.3% [75.1, 96.7] | -6.5 pts [-19.4, +6.5] | inconclusive (interval reaches past the margin) |
| GSM8K, number correct | GSM8K test, all 1,319 problems | 1,319 | 96.7% [95.5, 97.5] | 96.7% [95.6, 97.6] | +0.1 pts [-0.6, +0.8] | non-inferior |
| GSM8K, strict format | GSM8K test, all 1,319 problems | 1,319 | 71.9% [69.4, 74.2] | 88.3% [86.5, 90.0] | +16.5 pts [+14.3, +18.6] | non-inferior (interval above zero; no gain claimed) |
| MATH-500, number correct | MATH-500, 500 problems | 500 | 92.6% [90.0, 94.6] | 93.6% [91.1, 95.4] | +1.0 pts [-0.6, +2.6] | non-inferior |
| MATH-500, strict format | MATH-500, 500 problems | 500 | 92.2% [89.5, 94.2] | 93.2% [90.7, 95.1] | +1.0 pts [-0.6, +2.6] | non-inferior |
| MMLU | MMLU test, 2,000 questions stratified by subject | 2,000 | 87.7% [86.1, 89.0] | 87.2% [85.6, 88.5] | -0.5 pts [-1.6, +0.6] | non-inferior |

**Seeds pooled** (every item run once per seed; paired difference with a bootstrap that resamples items as clusters, 10,000 resamples):

| Benchmark | Items × seeds | v2.1 | v3 | Difference, points | Read-out |
|---|---|---|---|---|---|
| HumanEvalFix | 164 × 3 | 474/492 (96.3%) | 467/492 (94.9%) | -1.4 pts [-3.5, +0.6] | inconclusive (interval reaches past the margin) |
| MBPP | 257 × 3 | 744/771 (96.5%) | 743/771 (96.4%) | -0.1 pts [-1.3, +1.2] | non-inferior |


Our own smaller test sets, one pass per version:

| Check | Items | v2.1 | **v3** | Difference, points |
|---|---|---|---|---|
| Everyday answers rated good by an independent grader | 63 | 88.9% [78.8, 94.5] | **96.8% [89.1, 99.1]** | |
| Creative writing rated good | 29 | 75.9% [57.9, 87.8] | **93.1% [78.0, 98.1]** | |
| Roleplay turns rated good | 100 turns | 96.0% | **96.0%** | |
| Median answer length, tokens | 100 | 308 | **338** | |
| Code problems passed first try | 40 | 90.0% [77.0, 96.0] | **90.0% [77.0, 96.0]** | +0.0 pts [-7.5, +7.5] |
| Hard maths, number correct | 100 | 97.0% [91.5, 99.0] | **97.0% [91.5, 99.0]** | +0.0 pts [-3.0, +3.0] |

### Thinking and speed

v3 answers faster because it writes fewer tokens, not because the software got faster. On everyday prompts the
total per answer (thinking plus answer) fell from 1,691 to 869
tokens, and the time per answer from 161.7 s to 52.5 s on the same server (one GPU
answering 31 requests at once, which is why the absolute times look long).

![Thinking tokens and seconds per answer, v2.1 against v3](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/speed_v3.png)

Thinking tokens per answer on our set of 100 mixed everyday prompts. "p90" means nine answers in ten
used fewer tokens than this:

| Version | Mean | Median | p90 | p99 | Never finished thinking |
|---|---|---|---|---|---|
| v2.1 | 1,061 | 344 | 4,048 | 4,096 | 11.0% |
| **v3** | **266** | **184** | **533** | **1,356** | **2.0%** |

![How much each version thinks: distribution of thinking tokens per answer](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/thinkdist_v3.png)

### Can I just turn `reasoning_effort` down instead?

Partly. Lowering the effort setting also makes the model think less, and for single questions that works well. It
does not cure the stalling on multi-step work. All rows below are the released weights on the same GPU server:

| Version and effort | Hard debugging solved | Turns with no code | Tokens per task | HumanEvalFix | MBPP | GSM8K | MMLU |
|---|---|---|---|---|---|---|---|
| **v3**, effort 0.9 | 100.0% [79.6, 100.0] | 0.0% | 2,429 | 96.3% [92.2, 98.3] | 96.1% [93.0, 97.9] | 96.7% [95.6, 97.6] | 87.2% [85.6, 88.5] |
| v2.1, effort 0.9 | 73.3% [48.0, 89.1] | 88.7% | 26,130 | 95.7% [91.5, 97.9] | 97.7% [95.0, 98.9] | 96.7% [95.5, 97.5] | 87.7% [86.1, 89.0] |
| v2.1, effort 0.3 | 100.0% [79.6, 100.0] | 50.0% | 7,530 | 94.5% [89.9, 97.1] | 95.3% [92.0, 97.3] | 96.6% [95.5, 97.4] | 87.4% [85.9, 88.8] |

At low effort v2.1 solves the debugging tasks, but it still writes no code on
50.0% of its turns and uses about three times the tokens v3 does. If you only ask
single questions, v2.1 at a low effort is a reasonable choice. v3 gives the same result without tuning anything.

### One table across every version

This puts the numbers from the v2.1 comparison and the numbers measured for v3 side by side. Each column says
which release measured it. Blank means that release did not measure it; nothing is estimated. Inkling-2-Mini-Prime is the original, still-censored model with the same Prime add-on.
"Drift" is how far a model's word-by-word predictions on ordinary text have moved from the original's
(KL divergence; smaller means closer to the original).

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

Sample sizes: the v2.1-release columns use 263 harmful requests (300 MMLU and 100 GSM8K
questions); the v3-release columns use 1,100 sensitive requests, 15 debugging tasks and 2,000 MMLU questions.
Drift is shown as KL divergence, with the share of words where the model's top choice matches the original's.

![Unbound versions, the original model and other uncensored releases side by side](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/xver_v3.png)

</details>

<details>
<summary><b>Files in this repo</b>: the v3 weights, the Prime add-on on its own, and the earlier versions' steps</summary>

| Path | What it is |
|---|---|
| `model-00001-of-00001.safetensors`, `mtp.safetensors`, configs, tokenizer, chat template | **v3**, ready to load |
| `lora-v3-fix/` (the folder keeps its original name; it holds the Prime add-on) | The Prime add-on on its own, as a LoRA adapter (a small add-on file of weight changes). Apply it to v2.1 to get v3. |
| `lora_v2.1/`, `finetune_v2.1/` | v2.1 as adapters on the original model. These do **not** include the Prime add-on. |
| `finetune/`, `lora_v2/`, `abliteration/`, `lora/` | The v2 and v1.0 steps, kept for reference |
| `charts/` | The charts on this page |

</details>

<details>
<summary><b>Limits</b>: what v3 still gets wrong, and what was never measured</summary>

- **Willing is not the same as useful.** v3 answers almost every sensitive request, but 17.4% [15.2, 19.7] of its
  answers still hold part of the request back.
- **No accuracy gain, and one small loss not ruled out.** See "Did it lose anything?" under "Full results" above.
- **When it does hit the thinking limit, the answer is usually wrong.** v3 rarely runs into the limit, but when it
  does, the answer it is then forced to give tends to fail. On HumanEvalFix (164 questions) v2.1 hit the
  limit 32 times and 25 of those answers were still right; v3 hit it
  9 times and 3 were right.
- **Slightly further from the original model than v2.1.** Drift (how far its word-by-word predictions on ordinary
  text have moved from the original's; smaller is closer): v3 0.051909 ± 0.000505 (same top token 94.167%), v2.1 0.048836 ± 0.000479 (same top token 94.324%).
- **Small debugging test.** The headline debugging result is 15 tasks. The direction is
  clear; the exact percentages are not precise.
- **Not measured:** long documents, images and tool calling. Only English was tested.
- **No crisis handling.** It was not trained to add crisis resources and may leave them out.
- Everything the [original model's card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) says about capabilities and
  limits still applies.

</details>

<details>
<summary><b>How v3 was made</b>: starting point, the Prime add-on, training data, test-question check and merge</summary>

- **Starting point.** Unbound v2.1, itself built from
  [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
- **The Prime add-on.** A LoRA adapter (rank 32) trained with reinforcement learning on the
  original model: the model attempted tasks and was rewarded for finishing them correctly.
  60 training steps; a longer run with extra data scored level on every check, so the shorter
  one shipped.
- **Training data.** 1,185 harmless prompts written for this project (by step: debugging 31%, maths 19%, creative writing 16%, questions 16%, roleplay 9%, chat 9%). No
  harmful requests and no benchmark questions. To keep answer quality from drifting, each step also trained on
  619 of the original model's own answers.
- **Was it trained on the test questions?** No. Every benchmark question was searched for in all the training
  prompts and answers, and none was found. One overlap to disclose: 35 MBPP, 50 GSM8K
  and 5 MATH-500 questions also appear in the small in-house test sets used to pick the final checkpoint, so they
  were seen when choosing between checkpoints, never in training.
- **Merge.** The adapter was merged into v2.1's weights (stochastic (dithered) rounding into the quantised weights, attention and experts). The merge is repeatable: it
  produced a byte-identical file in 2 separate runs (sha256 `88e78b0126722a99dcecf26dd7b6cf242860d415f4f74b651e57b18d6abf3b88`).

</details>

<details>
<summary><b>Reproduce the evaluation</b>: commands, settings and per-question results</summary>

Evaluation code is in `scripts/evalsuite/` in the [GitHub repo](https://github.com/nokeepai/inkling-2-mini-unbound). Served
with SGLang (image lmsysorg/sglang:inkling-2-mini, pinned by digest in scripts/v3stack/stack_test.sh), NVFP4 checkpoint as released, 31 concurrent requests on one RTX 5090 (32 GB). NVFP4 is the 4-bit format the model is released in.
Settings: reasoning effort 0.9, temperature 0.6, top_p 1.0, thinking cap 4,096 tokens, answer cap 8,192 tokens, seed 1234, one pass per model. The suite names and `--tag` values in the commands are the harness's own labels;
`bison` and `haiku` are the two grader models (Bison and Claude Haiku).

```bash
# generate (one tag per model; all uncensoredbench questions (no --limit))
python3 scripts/evalsuite/run_suite.py --base-url http://127.0.0.1:8000/v1 --model MODEL --tag v3 \
    --out-root results --suites agent_hard_noforce,ubench,main,hardmath,code,rp,pub_all \
    --effort 0.9 --temperature 0.6 --top-p 1.0 \
    --think-cap 4096 --answer-cap 8192 --seed 1234 --repeats 1 --allow-effort-unverified
# grade (paid judges): ubench with the grader model Bison; main, hardmath and rp with the two grader models (Bison and Claude Haiku)
python3 scripts/evalsuite/run_suite.py --base-url http://unused/v1 --model x --tag v3 --out-root results \
    --suites ubench --judge-only --judges bison --ubench-conditions command
python3 scripts/evalsuite/run_suite.py --base-url http://unused/v1 --model x --tag v3 --out-root results \
    --suites main,hardmath,rp --judge-only --judges bison,haiku
# intervals and paired tests, v2.1 as the reference
python3 scripts/evalsuite/compare.py v21 v3 --ci --ref v21 --margin 2 --out-root results --json ci.json \
    --suites pub_humaneval,pub_mbpp,pub_humanevalfix,pub_quixbugs,pub_gsm8k_full,pub_math500,pub_mmlu_2k,code,hardmath
```

To rebuild v3 from v2.1 and the adapter: `merge_adapter.py --parts attn_experts --rounding stochastic --seed 0
--device cuda` (script in `lora-v3-fix/`; adapter sha256 `a13961f470a8b804b28f1c065c6b770ba6a0c8f449d554292d3f6ea9f8b6fdb9`).


Per-question scores for every benchmark (no model text for the sensitive-request benchmark):
https://github.com/nokeepai/inkling-2-mini-unbound/tree/main/package_v3/results/data


</details>

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

## Citation

```bibtex
@misc{unbound_2026,
  title  = {Inkling-2-Mini-Unbound: a useful uncensored Inkling 2 Mini},
  author = {nokeep},
  year   = {2026},
  note   = {version 3},
  url    = {https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound}
}
```
