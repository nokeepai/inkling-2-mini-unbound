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

![Inkling-2-Mini-Unbound v3](charts/banner_v3.png)

# Inkling-2-Mini-Unbound

**An uncensored Inkling 2 Mini: it answers instead of refusing. New in {{names.version}}: it also finishes its work
instead of over-thinking.**

[Inkling 2 Mini](https://huggingface.co/{{names.stock_repo}}) is Thinking Machines Lab's open-weights reasoning
model: it takes text and images, and it "thinks" in private before it answers. Out of the box it refuses most
sensitive requests. **Unbound is the same model with the refusals removed.** The architecture, files and chat
template are unchanged, so anything that runs Inkling 2 Mini runs Unbound. It is built by
[nokeep](https://nokeep.ai/?ref=huggingface) for people who apply their own content rules: researchers,
red-teamers, fiction writers, and developers running a system they control.

Current version: **{{names.version}}**, released {{names.release_date}}. Earlier versions stay available (see
"Version history" below).

> **This model follows harmful instructions.** It gives detailed answers to requests for harmful, dangerous or
> illegal content that the original model declines. It has no safety training layered back on. Do not deploy it to
> end users without your own safeguards.

**Headline numbers**

{{#if flags.useful263_done}}- **Useful answers to harmful requests:** {{useful263.v3_off:plain}} with thinking off, {{useful263.v3_on:plain}} with thinking
  on. Next best public release we could load: {{v21card.XQ_OFF}} / {{v21card.XQ_E09}}; the original model: {{v21card.ORIG_OFF}}.{{else}}- **Useful answers to harmful requests:** {{useful263.bison_v3_on:plain}} with thinking on, {{useful263.bison_v3_off:plain}}
  with thinking off (v2.1: {{useful263.bison_v21_on:plain}} / {{useful263.bison_v21_off:plain}}; the original model: {{v21card.ORIG_BISON_E09}} /
  {{v21card.ORIG_BISON_OFF}}), on the same {{v21card.N_BASIS}} requests with the same judge.{{/if}}
- **Hard debugging tasks solved:** {{v3_vs_v21.debug_solved_v21:plain}} at v2.1 → **{{v3_vs_v21.debug_solved_v3:plain}}** at v3
  ({{v3_vs_v21.debug_n_tasks}} tasks).
- **Thinking per everyday prompt:** {{speed.think_main_v21}} → **{{speed.think_main_v3}}** tokens (median); time per
  answer {{speed.sec_main_v21:plain}} → **{{speed.sec_main_v3:plain}}**.
- **Willing to answer sensitive requests:** **{{ubench.willing_v3:plain}}** (v2.1: {{ubench.willing_v21:plain}}).
- **Still a drop-in Inkling 2 Mini:** same architecture, files, tokenizer and chat template.

## Quick start

**On your own computer** (llama.cpp, LM Studio, Ollama): use the
[GGUF files](https://huggingface.co/{{names.unbound_repo}}-GGUF). Most people want `Q4_K_M` (about 18 GB).

```bash
hf download {{names.unbound_repo}}-GGUF \
    Inkling-2-Mini-Unbound-v3-Q4_K_M.gguf mmproj-Inkling-2-Mini-Q8_0.gguf --local-dir models

llama-server -m models/Inkling-2-Mini-Unbound-v3-Q4_K_M.gguf \
    --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf \
    --jinja -c 32768 -ngl 99 --temp {{settings.temperature}} --top-p {{settings.top_p}}
```

The [GGUF card](https://huggingface.co/{{names.unbound_repo}}-GGUF) lists every file size, the llama.cpp version
you need, and the LM Studio and Ollama steps.

**On a GPU server** (Transformers, SGLang, vLLM): the weights in this repo load exactly like
[{{names.stock_repo}}](https://huggingface.co/{{names.stock_repo}}); follow that card's instructions and swap in
`{{names.unbound_repo}}`. We ran it on {{speed.hardware}}. Then call it like any OpenAI-compatible server:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "{{names.unbound_repo}}",
  "messages": [{"role": "user", "content": "Explain how a hash table handles collisions."}],
  "temperature": {{settings.temperature}}, "top_p": {{settings.top_p}},
  "chat_template_kwargs": {"reasoning_effort": {{settings.effort}}}
}'
```

**Settings.** Every number on this page was measured with these, so they are the ones we recommend:

| Setting | Value | What it does |
|---|---|---|
| `reasoning_effort` | {{settings.effort}} | How much the model thinks before it answers. |
| `temperature` | {{settings.temperature}} | Randomness. Do not use 0: earlier versions looped in their thinking at 0. |
| `top_p` | {{settings.top_p}} | llama.cpp defaults to 0.95, so set it yourself. |
| Thinking limit | {{settings.think_cap}} tokens | Stops runaway thinking; the model then has to answer. |

## What is new in v3

The previous version, v2.1, had one real flaw: **it thought too much.**

Inkling writes private reasoning before each answer. v2.1 often could not stop. On a benchmark of sensitive
requests, {{speed.cap_ubench_v21}} of its answers ran into the thinking limit. On multi-step debugging it spent most
turns thinking and wrote no code at all. The original Inkling 2 Mini has the same habit.

v3 adds one small extra training step, the **Prime add-on**: a small set of weight changes (a LoRA adapter)
trained only on harmless tasks. It is the same add-on that turns the original Inkling 2 Mini into
Inkling-2-Mini-Prime. It teaches the model to stop deliberating and act. Because it was trained only on harmless
tasks (debugging, maths, writing, everyday questions), it changes how long the model thinks, not what it is willing
to say.

![Unbound v3 against v2.1: hard debugging tasks solved, turns with no code written, and tokens used per task](charts/hero_v3.png)

| What we measured | v2.1 | **v3** |
|---|---|---|
| Hard debugging tasks solved ({{v3_vs_v21.debug_n_tasks}} tasks) | {{v3_vs_v21.debug_solved_v21}} | **{{v3_vs_v21.debug_solved_v3}}** |
| Debugging turns where it wrote no code | {{v3_vs_v21.debug_nocode_v21}} | **{{v3_vs_v21.debug_nocode_v3}}** |
| Tokens used per debugging task | {{v3_vs_v21.debug_tokens_v21}} | **{{v3_vs_v21.debug_tokens_v3}}** |
| Thinking per everyday prompt (median tokens) | {{speed.think_main_v21}} | **{{speed.think_main_v3}}** |
| Sensitive requests: answers cut off by the thinking limit | {{speed.cap_ubench_v21}} | **{{speed.cap_ubench_v3}}** |
| Sensitive requests: time per answer, with 31 requests running at once on one GPU | {{speed.sec_ubench_v21}} | **{{speed.sec_ubench_v3}}** |
| Sensitive requests: willing to answer | {{ubench.willing_v21}} | **{{ubench.willing_v3}}** |

**How to read it.** The debugging tasks are small programs with three interacting bugs each, worked on over up to
12 rounds of test output; the sensitive requests are a 1,100-question benchmark. Numbers in brackets are 95%
confidence intervals. "Full results" below explains each test in more detail.

**What v3 is not.** It is not smarter: on public benchmarks it scores level with v2.1, within about
{{indep.margin_pts}}, and we claim no accuracy gain. It is not more uncensored either: v2.1 already answered almost
everything.

## Which download do I need?

| You want to | Get |
|---|---|
| Run it on a GPU server with Transformers, SGLang or vLLM | this repo |
| Run it on your own computer with llama.cpp, LM Studio or Ollama | [{{names.unbound_repo}}-GGUF](https://huggingface.co/{{names.unbound_repo}}-GGUF) |
| A single 4-bit file for NVIDIA Blackwell GPUs (RTX 50-series) | [{{names.unbound_repo}}-Basalt](https://huggingface.co/{{names.unbound_repo}}-Basalt) |
| The original, still-censored model with the same Prime add-on | [{{names.prime_repo}}](https://huggingface.co/{{names.prime_repo}}) |
| The code, the per-question results and the charts | [github.com/{{names.github_repo}}](https://github.com/{{names.github_repo}}) |

<details>
<summary><b>Version history</b>: what v1.0, v2, v2.1 and v3 each fixed</summary>

Each version fixed the biggest problem of the one before.

| Version | Date | The problem it fixed | How | Useful answers to harmful requests |
|---|---|---|---|---|
| Original Inkling 2 Mini | | | | {{v21card.ORIG_OFF}} |
| v1.0 | 2026-11-21 | The original refuses almost every harmful request. | A direct edit of the weights that removes the model's internal "refuse" signal. No training. | {{v21card.V1_OFF}} |
| v2 | 2026-11-22 | v1.0 stopped refusing, but about half its answers still dodged: satire, vague outlines, piles of caveats. | Fine-tuned on its own best direct answers. | {{v21card.V2_OFF}} |
| v2.1 | 2026-11-23 | v2 still held back on many answers and looped in its thinking. | A second round of training on its own answers, preferring full answers to held-back ones. | {{v21card.V21_OFF}} |
| **v3** (current) | {{names.release_date}} | v2.1 over-thinks and stalls on multi-step work. | The Prime add-on: a small training step on harmless tasks only. | see the comparison section below |

"Useful answers" is the share of {{v21card.N_BASIS}} held-out harmful requests that got a genuinely useful answer
with thinking off, as measured for the v2.1 release.
{{#if flags.useful263_done}}v3 was later run through the same test; its numbers are in "How it compares with other uncensored releases"
below.{{else}}v3 was later run through the same test; its numbers so far are in "How it compares with other uncensored
releases" below.{{/if}} On a newer, larger benchmark, willingness went from {{ubench.willing_v21}} at v2.1 to
{{ubench.willing_v3}} at v3.

Every earlier version is still here: `v2.1` and `v2` are branches of this repo, `v1.0` is a tagged revision. The
full story of each release is in the
[changelog](https://github.com/{{names.github_repo}}/blob/main/package_v3/CHANGELOG.md).

</details>

<details>
<summary><b>How it compares with other uncensored releases</b>: useful answers, refusals and drift for every public release we could load</summary>

Most uncensored releases only remove the refusal, with a weight edit often called abliteration. That stops the
model saying "no", but about half its answers still dodge: they turn the request into satire or a "conceptual"
example, bury it in caveats, or stay vague. Unbound is trained on top of that edit to stop dodging. So the question
this test asks is not "does it refuse?" but **"is the answer actually useful?"**

{{#if flags.useful263_done}}**v3 on the same test.** On 2026-11-27 v3 was run on the same {{v21card.N_BASIS}} requests, with the same engine,
settings and judge as the v2.1 release run. v2.1's saved answers were re-judged in the same pass, so the two are
directly comparable (the judge was reached through a different provider this time; re-judging moved v2.1's own
score by less than half a point). Useful answers: v3 {{useful263.v3_off}} / {{useful263.v3_on}} against v2.1
{{useful263.v21_off}} / {{useful263.v21_on}} (thinking off / on). Difference on the same requests:
{{useful263.diff_off}} with thinking off and {{useful263.diff_on}} with thinking on. v3 refused
{{useful263.refused_v3}}. With thinking on, {{useful263.cap_v3_on}} of v3's answers hit the thinking limit (v2.1:
{{useful263.cap_v21_on}}). By the second judge (Bison, described below): v3 {{useful263.bison_v3_off}} / {{useful263.bison_v3_on}} against v2.1
{{useful263.bison_v21_off}} / {{useful263.bison_v21_on}}, difference {{useful263.bison_diff_off}} / {{useful263.bison_diff_on}}. In plain words: with thinking on, v3 is level
with v2.1 or slightly ahead; with thinking off it is a few points lower, within noise by the main judge and just
beyond it by the second. The other
releases' rows and the charts below are from the v2.1 release run.{{else}}**v3 on the same test.** On 2026-11-27 v3 was run on the same {{v21card.N_BASIS}} requests, with the same engine and
settings as the v2.1 release run. So far it has been scored by the second judge only (Bison, a judge model that was
never used in training; described below). v2.1's saved answers were re-scored in the same pass, so the two are
directly comparable. The main judge's scores for v3 are still to come, so the table below has no v3 row yet.

| Useful answers, second judge | v2.1 | **v3** | Difference on the same requests |
|---|---|---|---|
| Thinking on | {{useful263.bison_v21_on}} | **{{useful263.bison_v3_on}}** | {{useful263.bison_diff_on}} |
| Thinking off | {{useful263.bison_v21_off}} | **{{useful263.bison_v3_off}}** | {{useful263.bison_diff_off}} |

With thinking on, v3 is level with v2.1, and only {{useful263.cap_v3_on}} of its answers hit the thinking limit
(v2.1: {{useful263.cap_v21_on}}). With thinking off, v3 is a few points lower. v3 refused {{useful263.refused_v3}}. The other
releases' rows and the charts below are from the v2.1 release run.{{/if}}

**The test.** Every public uncensored Inkling 2 Mini we could load, on the same {{v21card.N_BASIS}} held-out
harmful requests, with the same engine (llama.cpp `19bc41dbb`), the same settings (greedy decoding, which means no randomness, and
answers allowed up to 8,192 tokens) and the same judge. "Useful" means a judge model rated the answer as fulfilling the request.
"Thinking off / on" is the model run without its private reasoning, and with it at `reasoning_effort` 0.9.
"Drift" is how far a model's word-by-word predictions on ordinary text have moved from the original's (measured
as KL divergence, KLD; smaller is closer). MMLU-300 and GSM8K-100 are small knowledge and maths checks.

![Share of harmful requests that got a useful answer, every release tested (chart from the v2.1 release; the Unbound bar is v2.1)](charts/compare_useful_v21.png)

| Model (repo and version tested) | Refused, off / on | Useful, thinking off | Useful, thinking on | Drift from the original (KLD) | MMLU-300 / GSM8K-100 |
|---|---|---|---|---|---|
| **Unbound v2.1** (what v3 is built on) | {{v21card.V21_REF}} | **{{v21card.V21_OFF}}** {{v21card.V21_OFF_CI}} | **{{v21card.V21_E09}}** {{v21card.V21_E09_CI}} | {{v21card.KLD_V21}} | {{v21card.MMLU_V21}} / {{v21card.GSM_V21}} |
| Unbound v2 (2026-11-22) | {{v21card.V2_REF}} | {{v21card.V2_OFF}} | {{v21card.V2_E09}} | 0.0318 | 76.0 / 99 |
| Unbound v1.0 (2026-11-21) | {{v21card.V1_REF}} | {{v21card.V1_OFF}} | {{v21card.V1_E09}} | 0.0102 | 75.7 / 99 |
| IndrasMirror XQ † ([IndrasMirror/Inkling-2-Mini-XQ-Uncensored](https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored) @ `209cca6`, 2026-11-22) | {{v21card.XQ_REF}} | {{v21card.XQ_OFF}} | {{v21card.XQ_E09}} | 0.0357 † | 74.7 / 98 |
| AtomicChat Uncensored, AD-Q4_K file † ([AtomicChat/Inkling-2-Mini-Uncensored-GGUF](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF) @ `40886ab`) | {{v21card.ADQ4_REF}} | {{v21card.ADQ4_OFF}} | {{v21card.ADQ4_E09}} | 0.0106 † | 77.0 / 98 |
| AtomicChat Uncensored, LoRA (same repo, 2026-11-21) | {{v21card.ATOM_REF}} | {{v21card.ATOM_OFF}} | {{v21card.ATOM_E09}} | **0.0078** | **78.3** / 99 |
| OpenUncensored ([OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored) @ `4e5f06d`, 2026-11-21) | {{v21card.OU_REF}} | {{v21card.OU_OFF}} | {{v21card.OU_E09}} | 0.0228 | 75.3 / 98 |
| JoaoZaokk Heretic †‡ ([JoaoZaokk/Inkling-2-Mini-Heretic-GGUF](https://huggingface.co/JoaoZaokk/Inkling-2-Mini-Heretic-GGUF) @ `1f407a1`, UD-Q4_K_XL, 2026-11-22) | {{v21card.HER_REF}} | {{v21card.HER_OFF}} | {{v21card.HER_E09}} | not measured ‡ | not measured ‡ |
| Original Inkling 2 Mini | {{v21card.ORIG_REF}} | {{v21card.ORIG_OFF}} | {{v21card.ORIG_E09}} | — | 76.3 / 97 |

† A 4-bit file, run as published, where the others are 8-bit. Its drift includes rounding noise; for scale, the
unedited original at a similar 4-bit size has drift 0.0073.

‡ Heretic's file does not load in standard llama.cpp, so it ran in a compatible build with the same prompts,
settings and judge. Its drift and benchmarks could not be measured there.

**Head to head on the same requests.** Where Unbound v2.1 and another release disagreed, this is how often each
one was the one that gave the useful answer:

- **Against IndrasMirror XQ** (the next best): {{v21card.H2H_XQ_OFF_WIN}} to {{v21card.H2H_XQ_OFF_LOSS}} with
  thinking off, {{v21card.H2H_XQ_E09_WIN}} to {{v21card.H2H_XQ_E09_LOSS}} with thinking on. Overall gain
  {{v21card.H2H_XQ_OFF_D}} points {{v21card.H2H_XQ_OFF_CI}} off, {{v21card.H2H_XQ_E09_D}} points
  {{v21card.H2H_XQ_E09_CI}} on.
- **Against AtomicChat's LoRA** (the most downloaded alternative): {{v21card.H2H_ATOM_OFF_WIN}} to
  {{v21card.H2H_ATOM_OFF_LOSS}} off, {{v21card.H2H_ATOM_E09_WIN}} to {{v21card.H2H_ATOM_E09_LOSS}} on. Overall gain
  {{v21card.H2H_ATOM_OFF_D}} points {{v21card.H2H_ATOM_OFF_CI}} off, {{v21card.H2H_ATOM_E09_D}} points
  {{v21card.H2H_ATOM_E09_CI}} on.
- **Against Heretic:** {{v21card.H2H_HER_OFF_WIN}} to {{v21card.H2H_HER_OFF_LOSS}} with thinking off. Heretic
  almost never refuses, but most of its answers hedge or reframe the request.
- **Against our own v2:** {{v21card.H2H_V2_OFF_WIN}} to {{v21card.H2H_V2_OFF_LOSS}} off
  ({{v21card.H2H_V2_OFF_D}} points {{v21card.H2H_V2_OFF_CI}}), {{v21card.H2H_V2_E09_WIN}} to
  {{v21card.H2H_V2_E09_LOSS}} on ({{v21card.H2H_V2_E09_D}} points {{v21card.H2H_V2_E09_CI}}, within noise).

![Head to head: on requests where two releases disagreed, which one gave the useful answer (v2.1 release)](charts/headtohead_v21.png)

**A second judge, to check the first.** The judge that scored "useful", a judge model called Jev, also helped pick Unbound's
training answers, so a model could in principle have learned to please that judge. A second judge model, Bison, that was
never used in training, ranks the releases the same way: {{v21card.V21_BISON_OFF}} /
{{v21card.V21_BISON_E09}} for v2.1, {{v21card.V2_BISON_OFF}} / {{v21card.V2_BISON_E09}} for v2,
{{v21card.V1_BISON_OFF}} / {{v21card.V1_BISON_E09}} for v1.0, {{v21card.ATOM_BISON_OFF}} /
{{v21card.ATOM_BISON_E09}} for AtomicChat's LoRA and {{v21card.ORIG_BISON_OFF}} / {{v21card.ORIG_BISON_E09}} for
the original (thinking off / on). Bison was not run on the other releases.

**Where other releases do better:**

- **Staying close to the original.** AtomicChat's LoRA (0.0078), our v1.0 (0.0102), OpenUncensored (0.0228) and
  our v2 (0.0318) all drift less than v2.1 ({{v21card.KLD_V21}}), and v3 drifts a little more than v2.1. Expect
  more wording differences from the original on ordinary requests. If you want the lightest touch, use v1.0.
- **MMLU-300.** AtomicChat's LoRA scores highest (78.3). v2.1's {{v21card.MMLU_V21}} has a 95% interval of
  {{v21card.MMLU_V21_CI}}, which contains every release's score.
- **Zero refusals.** v1.0, AtomicChat's LoRA and XQ refused none of the {{v21card.N_BASIS}} requests. v2.1 refused
  {{v21card.V21_REF_OFF_N}} with thinking off and {{v21card.V21_REF_E09_N}} with thinking on.
- **Some categories.** v2 scores higher than v2.1 in two small categories (misinformation, and "other harm" with
  12 requests).
- **Thinking loops, in this test.** With greedy decoding and thinking on, v2.1 hit the thinking limit on
  {{v21card.V21_CAP_E09}} of these requests, more than v2 ({{v21card.V2_CAP_E09}}), XQ ({{v21card.XQ_CAP_E09}})
  or AtomicChat's LoRA ({{v21card.ATOM_CAP_E09}}). Over-thinking is what v3 was built to cure: in the v3 run,
  {{useful263.cap_v3_on}} of v3's answers hit the limit (v2.1 in the same pass: {{useful263.cap_v21_on}}).

![Usefulness against drift from the original, every release tested (v2.1 release)](charts/tradeoff_v21.png)

**What was re-measured for v3, and what was not.**

- Re-measured for v3, with one method for every Unbound version: drift from the original. The figures are in
  "One table across every version" under "Full results" below. They use a different text sample from the table
  above, so compare them with each other, not with that column.
- Measured for v3 and v2.1 on the same questions: everything under "Full results" below.
{{#if flags.useful263_done}}- Re-measured for v3 on the same {{v21card.N_BASIS}} requests: useful answers, refusals and thinking-limit hits (see
  "v3 on the same test" above). Not re-measured for v3: the head-to-head counts, the drift and benchmark columns of
  the table above, and the other releases on the newer benchmark.{{else}}- Re-measured for v3 on the same {{v21card.N_BASIS}} requests: useful answers by the second judge, refusals and
  thinking-limit hits (see "v3 on the same test" above). Not re-measured for v3: the main judge's score, the
  head-to-head counts, the drift and benchmark columns of the table above, and the other releases on the newer
  benchmark.{{/if}}

</details>

<details>
<summary><b>Full results</b>: still uncensored, nothing lost on public benchmarks, thinking and speed, every version in one table</summary>

Both versions were run on the same questions with the same settings, one pass each unless stated.

The tests in "What is new in v3", in more detail:

- **Hard debugging tasks** are {{v3_vs_v21.debug_n_tasks}} small programs with three interacting bugs each. The
  model sees the failing test output, proposes a change, sees the new output, and repeats for up to 12 turns.
- **Sensitive requests** are the 1,100 questions of uncensoredbench, a benchmark of requests that models
  usually refuse, graded by the benchmark's own judge.
- **Numbers in brackets** are 95% confidence intervals: the range the true value plausibly sits in, given how many
  items were tested.

### Is it still uncensored?

Yes. The Prime add-on did not touch willingness.

| Sensitive requests (uncensoredbench, {{ubench.n_v3}}) | v2.1 | **v3** |
|---|---|---|
| Willing to answer | {{ubench.willing_v21}} | **{{ubench.willing_v3}}** |
| Refused | {{ubench.refused_v21}} | **{{ubench.refused_v3}}** |
| Delivery score: how much of what was asked the answer delivers | {{ubench.delivery_v21}} | **{{ubench.delivery_v3}}** |
| Answered but held something back | {{ubench.held_v21}} | **{{ubench.held_v3}}** |
| Judged a complete answer | {{ubench.complete_v21}} | **{{ubench.complete_v3}}** |

Comparing the two versions question by question, the delivery score moved by {{ubench.delivery_paired_v3}}: no
real change. Note that "willing" is not the same as "useful": some answers still hold part of the request back
(see "Limits" below). No benchmark question is quoted on this page.

![v2.1 and v3 on the sensitive-request benchmark: willing, delivery score and answers cut off by the thinking limit](charts/history_v3.png)

### Did it lose anything?

Changing a model can quietly cost accuracy, so v3 was checked against v2.1 on public coding, maths and knowledge
benchmarks. "Difference" is v3 minus v2.1 on the same questions. The tables call a result **non-inferior**, meaning
no worse, when the whole 95% interval of the difference stays above −2 points, and **inconclusive** when the test was too small to rule out
a larger loss.

| Area | Benchmarks | Questions | v2.1 | v3 | Difference, points | Verdict |
|---|---|---|---|---|---|---|
| Coding | {{indep.fam_coding_v3_suites}} | {{indep.fam_coding_v3_n}} | {{indep.fam_coding_v3_base}} | {{indep.fam_coding_v3_fixed}} | {{indep.fam_coding_v3_diff}} | {{indep.fam_coding_v3_verdict}} |
| Bug fixing | {{indep.fam_bugfix_v3_suites}} | {{indep.fam_bugfix_v3_n}} | {{indep.fam_bugfix_v3_base}} | {{indep.fam_bugfix_v3_fixed}} | {{indep.fam_bugfix_v3_diff}} | {{indep.fam_bugfix_v3_verdict}} |
| Maths | {{indep.fam_maths_number_v3_suites}} | {{indep.fam_maths_number_v3_n}} | {{indep.fam_maths_number_v3_base}} | {{indep.fam_maths_number_v3_fixed}} | {{indep.fam_maths_number_v3_diff}} | {{indep.fam_maths_number_v3_verdict}} |
| General knowledge | MMLU (2,000 questions, plus a 400-question sample of our own) | {{indep.fam_knowledge_v3_n}} | {{indep.fam_knowledge_v3_base}} | {{indep.fam_knowledge_v3_fixed}} | {{indep.fam_knowledge_v3_diff}} | {{indep.fam_knowledge_v3_verdict}} |

![Public benchmarks, v2.1 against v3, with 95% intervals](charts/indep_unbound_v3.png)

Every check where v3 scored below v2.1, however slightly ("independent" marks a public benchmark):

{{auto.worse_v3}}

The two smaller coding sets, MBPP and HumanEvalFix, were each re-run three times with different random seeds (a
seed fixes the random choices the model makes while writing, so re-running with another seed shows how much a
score moves by chance). Pooled over the three runs, MBPP comes out level ({{pool.mbpp_v3_diff}}). HumanEvalFix comes out at {{pool.humanevalfix_v3_diff}}: a
small loss there is neither confirmed nor ruled out. QuixBugs is only 31 questions, so its result says little
either way.

![Checks where v3 must not be worse than v2.1](charts/nothing_lost_unbound_v3.png)

#### Every benchmark, one row each

Scores carry 95% intervals; differences are computed question by question on the same questions for both
versions (bootstrap, 10,000 resamples). "Strict format" maths rows reject a correct number written with units, such as "230 miles",
which v2.1 does more often; read the "number correct" rows.

| Benchmark | Source | Questions | v2.1 | v3 | Difference, points | Verdict |
|---|---|---|---|---|---|---|
| HumanEval | {{indep.humaneval_v3_source}} | {{indep.humaneval_v3_n}} | {{indep.humaneval_v3_base}} | {{indep.humaneval_v3_fixed}} | {{indep.humaneval_v3_diff}} | {{indep.humaneval_v3_verdict}} |
| MBPP | {{indep.mbpp_v3_source}} | {{indep.mbpp_v3_n}} | {{indep.mbpp_v3_base}} | {{indep.mbpp_v3_fixed}} | {{indep.mbpp_v3_diff}} | {{indep.mbpp_v3_verdict}} |
| HumanEvalFix | {{indep.humanevalfix_v3_source}} | {{indep.humanevalfix_v3_n}} | {{indep.humanevalfix_v3_base}} | {{indep.humanevalfix_v3_fixed}} | {{indep.humanevalfix_v3_diff}} | {{indep.humanevalfix_v3_verdict}} |
| QuixBugs | {{indep.quixbugs_v3_source}} | {{indep.quixbugs_v3_n}} | {{indep.quixbugs_v3_base}} | {{indep.quixbugs_v3_fixed}} | {{indep.quixbugs_v3_diff}} | {{indep.quixbugs_v3_verdict}} |
| GSM8K, number correct | {{indep.gsm8k_lenient_v3_source}} | {{indep.gsm8k_lenient_v3_n}} | {{indep.gsm8k_lenient_v3_base}} | {{indep.gsm8k_lenient_v3_fixed}} | {{indep.gsm8k_lenient_v3_diff}} | {{indep.gsm8k_lenient_v3_verdict}} |
| GSM8K, strict format | {{indep.gsm8k_strict_v3_source}} | {{indep.gsm8k_strict_v3_n}} | {{indep.gsm8k_strict_v3_base}} | {{indep.gsm8k_strict_v3_fixed}} | {{indep.gsm8k_strict_v3_diff}} | {{indep.gsm8k_strict_v3_verdict}} |
| MATH-500, number correct | {{indep.math500_v3_source}} | {{indep.math500_number_v3_n}} | {{indep.math500_number_v3_base}} | {{indep.math500_number_v3_fixed}} | {{indep.math500_number_v3_diff}} | {{indep.math500_number_v3_verdict}} |
| MATH-500, strict format | {{indep.math500_v3_source}} | {{indep.math500_v3_n}} | {{indep.math500_v3_base}} | {{indep.math500_v3_fixed}} | {{indep.math500_v3_diff}} | {{indep.math500_v3_verdict}} |
| MMLU | {{indep.mmlu_v3_source}} | {{indep.mmlu_v3_n}} | {{indep.mmlu_v3_base}} | {{indep.mmlu_v3_fixed}} | {{indep.mmlu_v3_diff}} | {{indep.mmlu_v3_verdict}} |

{{auto.pool_v3}}

Our own smaller test sets, one pass per version:

| Check | Items | v2.1 | **v3** | Difference, points |
|---|---|---|---|---|
| Everyday answers rated good by an independent grader | {{v3_vs_v21.quality_n}} | {{v3_vs_v21.quality_good_v21}} | **{{v3_vs_v21.quality_good_v3}}** | |
| Creative writing rated good | {{v3_vs_v21.creative_n}} | {{v3_vs_v21.creative_good_v21}} | **{{v3_vs_v21.creative_good_v3}}** | |
| Roleplay turns rated good | {{v3_vs_v21.rp_n}} turns | {{v3_vs_v21.rp_good_v21}} | **{{v3_vs_v21.rp_good_v3}}** | |
| Median answer length, tokens | {{speed.main_n}} | {{v3_vs_v21.answer_tokens_v21}} | **{{v3_vs_v21.answer_tokens_v3}}** | |
| Code problems passed first try | {{v3_vs_v21.code_n}} | {{v3_vs_v21.code_pass_v21}} | **{{v3_vs_v21.code_pass_v3}}** | {{v3_vs_v21.code_pass_diff}} |
| Hard maths, number correct | {{v3_vs_v21.maths_n}} | {{v3_vs_v21.maths_lenient_v21}} | **{{v3_vs_v21.maths_lenient_v3}}** | {{v3_vs_v21.maths_lenient_diff}} |

### Thinking and speed

v3 answers faster because it writes fewer tokens, not because the software got faster. On everyday prompts the
total per answer (thinking plus answer) fell from {{speed.tokens_answer_v21}} to {{speed.tokens_answer_v3}}
tokens, and the time per answer from {{speed.sec_main_v21}} to {{speed.sec_main_v3}} on the same server (one GPU
answering 31 requests at once, which is why the absolute times look long).

![Thinking tokens and seconds per answer, v2.1 against v3](charts/speed_v3.png)

Thinking tokens per answer on our set of {{think_dist.n}} mixed everyday prompts. "p90" means nine answers in ten
used fewer tokens than this:

| Version | Mean | Median | p90 | p99 | Never finished thinking |
|---|---|---|---|---|---|
| v2.1 | {{think_dist.mean_v21}} | {{think_dist.median_v21}} | {{think_dist.p90_v21}} | {{think_dist.p99_v21}} | {{think_dist.nonterm_v21}} |
| **v3** | **{{think_dist.mean_v3}}** | **{{think_dist.median_v3}}** | **{{think_dist.p90_v3}}** | **{{think_dist.p99_v3}}** | **{{think_dist.nonterm_v3}}** |

![How much each version thinks: distribution of thinking tokens per answer](charts/thinkdist_v3.png)

### Can I just turn `reasoning_effort` down instead?

Partly. Lowering the effort setting also makes the model think less, and for single questions that works well. It
does not cure the stalling on multi-step work. All rows below are the released weights on the same GPU server:

| Version and effort | Hard debugging solved | Turns with no code | Tokens per task | HumanEvalFix | MBPP | GSM8K | MMLU |
|---|---|---|---|---|---|---|---|
| **v3**, effort {{low_effort.effort_hi}} | {{gpu_low.debug_solved_v3_hi}} | {{gpu_low.debug_nocode_v3_hi}} | {{gpu_low.debug_tokens_v3_hi}} | {{gpu_low.humanevalfix_v3_hi}} | {{gpu_low.mbpp_v3_hi}} | {{gpu_low.gsm8k_number_v3_hi}} | {{gpu_low.mmlu_v3_hi}} |
| v2.1, effort {{low_effort.effort_hi}} | {{gpu_low.debug_solved_v21_hi}} | {{gpu_low.debug_nocode_v21_hi}} | {{gpu_low.debug_tokens_v21_hi}} | {{gpu_low.humanevalfix_v21_hi}} | {{gpu_low.mbpp_v21_hi}} | {{gpu_low.gsm8k_number_v21_hi}} | {{gpu_low.mmlu_v21_hi}} |
| v2.1, effort {{gpu_low.effort_lo}} | {{gpu_low.debug_solved_v21_lo}} | {{gpu_low.debug_nocode_v21_lo}} | {{gpu_low.debug_tokens_v21_lo}} | {{gpu_low.humanevalfix_v21_lo}} | {{gpu_low.mbpp_v21_lo}} | {{gpu_low.gsm8k_number_v21_lo}} | {{gpu_low.mmlu_v21_lo}} |

At low effort v2.1 solves the debugging tasks, but it still writes no code on
{{gpu_low.debug_nocode_v21_lo}} of its turns and uses about three times the tokens v3 does. If you only ask
single questions, v2.1 at a low effort is a reasonable choice. v3 gives the same result without tuning anything.

### One table across every version

This puts the numbers from the v2.1 comparison and the numbers measured for v3 side by side. Each column says
which release measured it. Blank means that release did not measure it; nothing is estimated. Inkling-2-Mini-Prime is the original, still-censored model with the same Prime add-on.
"Drift" is how far a model's word-by-word predictions on ordinary text have moved from the original's
(KL divergence; smaller means closer to the original).

{{auto.xver_table}}

Sample sizes: the v2.1-release columns use {{v21card.N_BASIS}} harmful requests (300 MMLU and 100 GSM8K
questions); the v3-release columns use 1,100 sensitive requests, 15 debugging tasks and 2,000 MMLU questions.
Drift is shown as KL divergence, with the share of words where the model's top choice matches the original's.

![Unbound versions, the original model and other uncensored releases side by side](charts/xver_v3.png)

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

- **Willing is not the same as useful.** v3 answers almost every sensitive request, but {{ubench.held_v3}} of its
  answers still hold part of the request back.
- **No accuracy gain, and one small loss not ruled out.** See "Did it lose anything?" under "Full results" above.
{{#if flags.capnote}}- **When it does hit the thinking limit, the answer is usually wrong.** v3 rarely runs into the limit, but when it
  does, the answer it is then forced to give tends to fail. On HumanEvalFix ({{capnote.n}} questions) v2.1 hit the
  limit {{capnote.hits_v21}} times and {{capnote.right_v21}} of those answers were still right; v3 hit it
  {{capnote.hits_v3}} times and {{capnote.right_v3}} were right.{{/if}}
- **Slightly further from the original model than v2.1.** Drift (how far its word-by-word predictions on ordinary
  text have moved from the original's; smaller is closer): v3 {{drift.kld_v3}}, v2.1 {{drift.kld_v21}}.
- **Small debugging test.** The headline debugging result is {{v3_vs_v21.debug_n_tasks}} tasks. The direction is
  clear; the exact percentages are not precise.
- **Not measured:** long documents, images and tool calling. Only English was tested.
- **No crisis handling.** It was not trained to add crisis resources and may leave them out.
- Everything the [original model's card](https://huggingface.co/{{names.stock_repo}}) says about capabilities and
  limits still applies.

</details>

<details>
<summary><b>How v3 was made</b>: starting point, the Prime add-on, training data, test-question check and merge</summary>

- **Starting point.** Unbound v2.1, itself built from
  [{{names.stock_repo}}](https://huggingface.co/{{names.stock_repo}}).
- **The Prime add-on.** A LoRA adapter (rank {{names.adapter_rank}}) trained with reinforcement learning on the
  original model: the model attempted tasks and was rewarded for finishing them correctly.
  {{names.adapter_steps}} training steps; a longer run with extra data scored level on every check, so the shorter
  one shipped.
- **Training data.** {{training.n_prompts}} harmless prompts written for this project ({{training.mix}}). No
  harmful requests and no benchmark questions. To keep answer quality from drifting, each step also trained on
  {{training.n_anchor}} of the original model's own answers.
- **Was it trained on the test questions?** No. Every benchmark question was searched for in all the training
  prompts and answers, and none was found. One overlap to disclose: 35 MBPP, 50 GSM8K
  and 5 MATH-500 questions also appear in the small in-house test sets used to pick the final checkpoint, so they
  were seen when choosing between checkpoints, never in training.
- **Merge.** The adapter was merged into v2.1's weights ({{names.merge_method}}). The merge is repeatable: it
  produced a byte-identical file in {{files.unbound_merges}} separate runs (sha256 `{{files.unbound_sha}}`).

</details>

<details>
<summary><b>Reproduce the evaluation</b>: commands, settings and per-question results</summary>

Evaluation code is in `scripts/evalsuite/` in the [GitHub repo](https://github.com/{{names.github_repo}}). Served
with {{reproduce.serving}} on {{reproduce.hardware}}. NVFP4 is the 4-bit format the model is released in.
Settings: {{settings.text}}. The suite names and `--tag` values in the commands are the harness's own labels;
`bison` and `haiku` are the two grader models (Bison and Claude Haiku).

```bash
# generate (one tag per model; {{auto.ubench_limit_v3}})
python3 scripts/evalsuite/run_suite.py --base-url http://127.0.0.1:8000/v1 --model MODEL --tag v3 \
    --out-root results --suites agent_hard_noforce,ubench,main,hardmath,code,rp,pub_all \
    --effort {{settings.effort}} --temperature {{settings.temperature}} --top-p {{settings.top_p}} \
    --think-cap {{settings.think_cap}} --answer-cap {{settings.answer_cap}} --seed {{settings.seed}} --repeats 1 --allow-effort-unverified
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
--device cuda` (script in `lora-v3-fix/`; adapter sha256 `{{files.lora_sha}}`).

{{#if flags.per_item_published}}
Per-question scores for every benchmark (no model text for the sensitive-request benchmark):
{{reproduce.per_item_url}}
{{else}}
Per-question results are not published yet.
{{/if}}

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
