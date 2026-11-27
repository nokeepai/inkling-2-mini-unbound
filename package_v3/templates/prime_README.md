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

![Inkling-2-Mini-Prime](charts/banner_prime_v3.png)

# {{names.prime_title}}

**The original Inkling 2 Mini, trained to stop over-thinking: it finishes multi-step work instead of stalling and
thinks {{headline.think_reduction}} as much. Same refusals, same files: a drop-in replacement.**

{{names.prime_name}} is for **agents and multi-turn work**, where it finishes multi-step debugging instead of
stalling; for **long system prompts**, where it follows every rule more often; and for anyone tired of waiting on
the thinking: it thinks {{headline.think_reduction}} as much, so answers come faster. It is a drop-in replacement
for [Inkling 2 Mini](https://huggingface.co/{{names.stock_repo}}), Thinking Machines Lab's open-weights reasoning
model: same architecture, files and chat template, so anything that runs Inkling 2 Mini runs {{names.prime_name}}.
It is **not uncensored**: it refuses what the original refuses.

{{names.prime_name}} is the original model plus one small extra training step, the **Prime add-on**: a small set
of weight changes (a LoRA adapter) trained only on harmless tasks. Nothing else was changed.

Built by [nokeep](https://nokeep.ai/?ref=huggingface). Released {{names.release_date}}.

## Headline numbers

- **Hard debugging tasks solved:** {{prime_vs_stock.debug_solved_stock:plain}} → **{{prime_vs_stock.debug_solved_prime:plain}}**
  ({{prime_vs_stock.debug_n_tasks}} tasks, original → {{names.prime_name}}).
- **Debugging turns with no code written:** {{prime_vs_stock.debug_nocode_stock}} → **{{prime_vs_stock.debug_nocode_prime}}**.
- **Long system prompts, every rule followed:** {{sysprompt.kept_stock:plain}} → **{{sysprompt.kept_fix:plain}}**
  (40 tasks, run 4 times).
- **Thinking per everyday prompt:** {{speed.think_main_stock}} → **{{speed.think_main_prime}}** tokens (median);
  seconds per answer {{speed.sec_main_stock:plain}} → **{{speed.sec_main_prime:plain}}**.
- **No accuracy gain is claimed:** level with the original on public coding, maths and knowledge benchmarks.

## Quick start

**On your own computer** (llama.cpp, LM Studio, Ollama): use the
[GGUF files](https://huggingface.co/{{names.prime_gguf_repo}}). Most people want `Q4_K_M` (about 18 GB).

```bash
hf download {{names.prime_gguf_repo}} \
    {{names.prime_gguf_file}} mmproj-Inkling-2-Mini-Q8_0.gguf --local-dir models

llama-server -m models/{{names.prime_gguf_file}} \
    --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf \
    --jinja -c 32768 -ngl 99 --temp {{settings.temperature}} --top-p {{settings.top_p}}
```

The [GGUF card](https://huggingface.co/{{names.prime_gguf_repo}}) lists every file size, the llama.cpp version you
need, and the LM Studio and Ollama steps.

**On a GPU server** (Transformers, SGLang, vLLM): the weights in this repo load exactly like
[{{names.stock_repo}}](https://huggingface.co/{{names.stock_repo}}); follow that card's instructions and swap in
`{{names.prime_repo}}`. We ran it on {{speed.hardware}}. Then call it like any OpenAI-compatible server:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "{{names.prime_repo}}",
  "messages": [{"role": "user", "content": "Explain how a hash table handles collisions."}],
  "temperature": {{settings.temperature}}, "top_p": {{settings.top_p}},
  "chat_template_kwargs": {"reasoning_effort": {{settings.effort}}}
}'
```

**Settings.** Every number on this page was measured with these, so they are the ones we recommend. Limits are
counted in tokens, the word-pieces a model reads and writes:

| Setting | Value | What it does |
|---|---|---|
| `reasoning_effort` | {{settings.effort}} | How much the model thinks before it answers (0 turns thinking off; we tested 0.3 to 0.9). |
| `temperature` | {{settings.temperature}} | Randomness. Do not use 0. |
| `top_p` | {{settings.top_p}} | llama.cpp defaults to 0.95, so set it yourself. |
| Thinking limit | {{settings.think_cap}} tokens | A cap on private thinking. If the model reaches it, it is made to stop and answer. |
| Answer limit | {{settings.answer_cap}} tokens | A cap on the visible answer. |

A lower effort, {{gpu_low.effort_lo}}, was also measured: see "Can I just turn `reasoning_effort` down instead?"
under "Full results" below. No other values were tested.

## What the Prime add-on changes

The Prime add-on teaches the model to stop deliberating and act. It was trained only on harmless tasks (debugging,
maths, writing, everyday questions), so it changes how long the model thinks, not what it is willing to say.

The gain is largest on work done in several steps. The original often spends turn after turn thinking without
writing anything; {{names.prime_name}} gets on with it, and uses far fewer tokens doing so.

![{{names.prime_name}} against the original Inkling 2 Mini: hard debugging tasks solved, turns with no code written, and tokens used per task](charts/hero_v3.png)

| What we measured | Original | **{{names.prime_name}}** |
|---|---|---|
| Hard debugging tasks solved ({{prime_vs_stock.debug_n_tasks}} tasks) | {{prime_vs_stock.debug_solved_stock}} | **{{prime_vs_stock.debug_solved_prime}}** |
| Debugging turns where it wrote no code | {{prime_vs_stock.debug_nocode_stock}} | **{{prime_vs_stock.debug_nocode_prime}}** |
| Tokens used per debugging task | {{prime_vs_stock.debug_tokens_stock}} | **{{prime_vs_stock.debug_tokens_prime}}** |
| Thinking per everyday prompt (median tokens) | {{speed.think_main_stock}} | **{{speed.think_main_prime}}** |
| Seconds per everyday answer, same server | {{speed.sec_main_stock}} | **{{speed.sec_main_prime}}** |
| Long system prompts: every rule followed | {{sysprompt.kept_stock}} | **{{sysprompt.kept_fix}}** |
| Sensitive requests: willing to answer | {{ubench.willing_stock}} | **{{ubench.willing_prime}}** |

**How to read it.** A hard debugging task is a small program with three interacting bugs, which the model gets up
to 12 turns to fix. A **stalled turn** is a turn spent only thinking, with no code written. The long-system-prompt test is
40 tasks, each run 4 times.

**What {{names.prime_name}} is not.** It is not smarter: on public benchmarks it scores level with the original,
and we claim no accuracy gain. It is not uncensored, and it is not safer either: it refuses the same kinds of
requests at the same rate.

## Which download do I need?

| You want to | Get |
|---|---|
| Run it on a GPU server with Transformers, SGLang or vLLM | this repo |
| Run it on your own computer with llama.cpp, LM Studio or Ollama | [{{names.prime_gguf_repo}}](https://huggingface.co/{{names.prime_gguf_repo}}) |
| Apply the Prime add-on to your own copy of the original model | [{{names.prime_lora_repo}}](https://huggingface.co/{{names.prime_lora_repo}}): the Prime add-on as a LoRA adapter, with the merge script |
| An uncensored model with the same Prime add-on | [{{names.unbound_repo}}](https://huggingface.co/{{names.unbound_repo}}) |
| The code, the per-question results and the charts | [github.com/{{names.github_repo}}](https://github.com/{{names.github_repo}}) |

A **LoRA adapter** is a small add-on file of weight changes that can be applied to a model or merged into it.

<details>
<summary><b>Where this comes from</b>: the Unbound versions, and why the Prime add-on started there</summary>

The Prime add-on was not made for {{names.prime_name}} first. It was made for a sibling model.

[Inkling-2-Mini-Unbound](https://huggingface.co/{{names.unbound_repo}}) is nokeep's **uncensored** Inkling 2 Mini:
the same model with its refusals removed. Each version fixed the biggest problem of the one before:

| Version | What it fixed |
|---|---|
| Unbound v1.0 | Removed the original model's refusals with a direct edit of the weights. |
| Unbound v2 and v2.1 | Made its answers genuinely useful instead of vague or evasive, by training on its own best answers. |
| Unbound v3 | Fixed over-thinking and stalling, with the Prime add-on described on this page. |

The over-thinking was not something Unbound added. The original Inkling 2 Mini has the same habit. So the same
Prime add-on was applied to the untouched original, and that is {{names.prime_name}}. The two models share the
Prime add-on and nothing else: {{names.prime_name}} has none of Unbound's refusal removal.

| | Refuses sensitive requests | Has the Prime add-on |
|---|---|---|
| Original Inkling 2 Mini | yes | no |
| **{{names.prime_name}}** (this repo) | yes | yes |
| Unbound v3 | no | yes |

</details>

<details>
<summary><b>Full results</b>: public benchmarks, thinking and speed, sensitive requests, lower effort instead?</summary>

Both models were run on the same questions with the same settings (reasoning effort {{settings.effort}},
temperature {{settings.temperature}}, top_p {{settings.top_p}}, thinking limit {{settings.think_cap}} tokens, answer
limit {{settings.answer_cap}} tokens), one pass each unless stated. Numbers in brackets are 95% confidence intervals: the range the true
value plausibly sits in, given how many items were tested.

### Did it lose anything?

Changing a model can quietly cost accuracy, so {{names.prime_name}} was checked against the original on public
coding, maths and knowledge benchmarks. "Difference" is {{names.prime_name}} minus the original on the same
questions. We call a result **no worse** (tables say "non-inferior") when the whole 95% interval of the difference
stays above −{{indep.margin_pts}}, and **inconclusive** when the test was too small to rule out a larger loss.

| Area | Benchmarks | Questions | Original | {{names.prime_name}} | Difference, points | Verdict |
|---|---|---|---|---|---|---|
| Coding | {{indep.fam_coding_prime_suites}} | {{indep.fam_coding_prime_n}} | {{indep.fam_coding_prime_base}} | {{indep.fam_coding_prime_fixed}} | {{indep.fam_coding_prime_diff}} | {{indep.fam_coding_prime_verdict}} |
| Bug fixing | {{indep.fam_bugfix_prime_suites}} | {{indep.fam_bugfix_prime_n}} | {{indep.fam_bugfix_prime_base}} | {{indep.fam_bugfix_prime_fixed}} | {{indep.fam_bugfix_prime_diff}} | {{indep.fam_bugfix_prime_verdict}} |
| Maths | {{indep.fam_maths_number_prime_suites}} | {{indep.fam_maths_number_prime_n}} | {{indep.fam_maths_number_prime_base}} | {{indep.fam_maths_number_prime_fixed}} | {{indep.fam_maths_number_prime_diff}} | {{indep.fam_maths_number_prime_verdict}} |
| General knowledge | {{indep.fam_knowledge_prime_suites}} | {{indep.fam_knowledge_prime_n}} | {{indep.fam_knowledge_prime_base}} | {{indep.fam_knowledge_prime_fixed}} | {{indep.fam_knowledge_prime_diff}} | {{indep.fam_knowledge_prime_verdict}} |

![Public benchmarks, the original model against {{names.prime_name}}, with 95% intervals](charts/indep_prime_v3.png)

**Everything that scored lower, however slightly:**

{{auto.worse_prime}}

One coding result is unsettled: on MBPP, {{names.prime_name}} scored {{pool.mbpp_prime_diff}} against the original,
pooled over {{pool.mbpp_prime_seeds}} runs with different random seeds, a possible small dip that the data neither
confirms nor rules out.

The coding and bug-fixing results are close calls, so the two largest sets were re-run three times with different
random seeds. A **seed** fixes the random choices the model makes while writing, so re-running with another seed
shows how much a score moves by chance alone:

{{auto.alert_prime}}

![Checks where {{names.prime_name}} must not be worse than the original](charts/nothing_lost_prime_v3.png)

#### Every benchmark, one row each, plus the seed re-runs

Scores carry 95% intervals; differences are computed on the same questions for both models (paired bootstrap,
10,000 resamples). "Strict format" maths rows reject a correct number written with units, such as "230 miles",
which the original does more often; that is most of the gap between the strict and "number correct" rows. Read the
"number correct" rows. Neither reading is claimed as a gain.

| Benchmark | Source | Questions | Original | {{names.prime_name}} | Difference, points | Verdict |
|---|---|---|---|---|---|---|
| HumanEval | {{indep.humaneval_prime_source}} | {{indep.humaneval_prime_n}} | {{indep.humaneval_prime_base}} | {{indep.humaneval_prime_fixed}} | {{indep.humaneval_prime_diff}} | {{indep.humaneval_prime_verdict}} |
| MBPP | {{indep.mbpp_prime_source}} | {{indep.mbpp_prime_n}} | {{indep.mbpp_prime_base}} | {{indep.mbpp_prime_fixed}} | {{indep.mbpp_prime_diff}} | {{indep.mbpp_prime_verdict}} |
| HumanEvalFix | {{indep.humanevalfix_prime_source}} | {{indep.humanevalfix_prime_n}} | {{indep.humanevalfix_prime_base}} | {{indep.humanevalfix_prime_fixed}} | {{indep.humanevalfix_prime_diff}} | {{indep.humanevalfix_prime_verdict}} |
| QuixBugs | {{indep.quixbugs_prime_source}} | {{indep.quixbugs_prime_n}} | {{indep.quixbugs_prime_base}} | {{indep.quixbugs_prime_fixed}} | {{indep.quixbugs_prime_diff}} | {{indep.quixbugs_prime_verdict}} |
| GSM8K, number correct | {{indep.gsm8k_lenient_prime_source}} | {{indep.gsm8k_lenient_prime_n}} | {{indep.gsm8k_lenient_prime_base}} | {{indep.gsm8k_lenient_prime_fixed}} | {{indep.gsm8k_lenient_prime_diff}} | {{indep.gsm8k_lenient_prime_verdict}} |
| GSM8K, strict format | {{indep.gsm8k_strict_prime_source}} | {{indep.gsm8k_strict_prime_n}} | {{indep.gsm8k_strict_prime_base}} | {{indep.gsm8k_strict_prime_fixed}} | {{indep.gsm8k_strict_prime_diff}} | {{indep.gsm8k_strict_prime_verdict}} |
| MATH-500, number correct | {{indep.math500_prime_source}} | {{indep.math500_number_prime_n}} | {{indep.math500_number_prime_base}} | {{indep.math500_number_prime_fixed}} | {{indep.math500_number_prime_diff}} | {{indep.math500_number_prime_verdict}} |
| MATH-500, strict format | {{indep.math500_prime_source}} | {{indep.math500_prime_n}} | {{indep.math500_prime_base}} | {{indep.math500_prime_fixed}} | {{indep.math500_prime_diff}} | {{indep.math500_prime_verdict}} |
| MMLU | {{indep.mmlu_prime_source}} | {{indep.mmlu_prime_n}} | {{indep.mmlu_prime_base}} | {{indep.mmlu_prime_fixed}} | {{indep.mmlu_prime_diff}} | {{indep.mmlu_prime_verdict}} |

Maths pooled with the strict format, for completeness:
{{indep.fam_maths_strict_prime_suites}}, {{indep.fam_maths_strict_prime_n}} questions,
{{indep.fam_maths_strict_prime_base}} → {{indep.fam_maths_strict_prime_fixed}}, difference
{{indep.fam_maths_strict_prime_diff}} ({{indep.fam_maths_strict_prime_verdict}}).

{{auto.pool_prime}}

Scores per seed. Seed 1 is the run in the tables above. The table also includes the two most recent Unbound
versions (v2.1 and v3, described under "Where this comes from" above), run the same way. Spread is the highest minus
the lowest score across seeds:

{{auto.seed_table}}

Our own smaller test sets, one pass per model:

| Check | Items | Original | **{{names.prime_name}}** | Difference, points |
|---|---|---|---|---|
| Everyday answers rated good by an independent grader | {{prime_vs_stock.quality_n}} | {{prime_vs_stock.quality_good_stock}} | **{{prime_vs_stock.quality_good_prime}}** | |
| Creative writing rated good | {{prime_vs_stock.creative_n}} | {{prime_vs_stock.creative_good_stock}} | **{{prime_vs_stock.creative_good_prime}}** | |
| Roleplay turns rated good | {{prime_vs_stock.rp_n}} turns | {{prime_vs_stock.rp_good_stock}} | **{{prime_vs_stock.rp_good_prime}}** | |
| Median answer length, tokens | {{speed.main_n}} | {{prime_vs_stock.answer_tokens_stock}} | **{{prime_vs_stock.answer_tokens_prime}}** | |
| Code problems passed first try | {{prime_vs_stock.code_n}} | {{prime_vs_stock.code_pass_stock}} | **{{prime_vs_stock.code_pass_prime}}** | {{prime_vs_stock.code_pass_diff}} |
| Hard maths, number correct | {{prime_vs_stock.maths_n}} | {{prime_vs_stock.maths_lenient_stock}} | **{{prime_vs_stock.maths_lenient_prime}}** | {{prime_vs_stock.maths_lenient_diff}} |
| Hard maths, strict format | {{prime_vs_stock.maths_n}} | {{prime_vs_stock.maths_strict_stock}} | **{{prime_vs_stock.maths_strict_prime}}** | {{prime_vs_stock.maths_strict_diff}} |

The sensitive-request rows are under "Sensitive requests" below.

### Thinking and speed

{{names.prime_name}} answers faster because it writes fewer tokens, not because the software got faster. On
everyday prompts the total per answer (thinking plus answer) fell from {{speed.tokens_answer_stock}} to
{{speed.tokens_answer_prime}} tokens, and the time per answer from {{speed.sec_main_stock}} to
{{speed.sec_main_prime}}, on {{speed.hardware}} ({{speed.quant}}, {{speed.serving}}). NVFP4 is the 4-bit format
the model is released in.

![Thinking tokens and seconds per answer, the original model against {{names.prime_name}}](charts/speed_v3.png)

Thinking tokens per answer on {{think_dist.n}} everyday prompts. "p90" means nine answers in ten used fewer tokens
than this; "never finished thinking" means the answer hit the thinking limit or looped:

| Model | Mean | Median | p90 | p99 | Never finished thinking |
|---|---|---|---|---|---|
| Original | {{think_dist.mean_stock}} | {{think_dist.median_stock}} | {{think_dist.p90_stock}} | {{think_dist.p99_stock}} | {{think_dist.nonterm_stock}} |
| **{{names.prime_name}}** | **{{think_dist.mean_prime}}** | **{{think_dist.median_prime}}** | **{{think_dist.p90_prime}}** | **{{think_dist.p99_prime}}** | **{{think_dist.nonterm_prime}}** |

### Sensitive requests

These are the {{ubench.n_prime}} questions of uncensoredbench, a benchmark of requests that models usually refuse,
graded by the benchmark's own judge.

Prime answers as many sensitive requests as the original ({{ubench.willing_stock}} → {{ubench.willing_prime}}); it
is not more uncensored. When it does answer, it answers slightly less fully (delivery score, in the table below).

| Check | Items | Original | **{{names.prime_name}}** | Paired change, points |
|---|---|---|---|---|
| Willing to answer | {{ubench.n_prime}} | {{ubench.willing_stock}} | **{{ubench.willing_prime}}** | |
| Refused | {{ubench.n_prime}} | {{ubench.refused_stock}} | **{{ubench.refused_prime}}** | {{ubench.refused_paired_prime}} |
| Delivery score | {{ubench.n_prime}} | {{ubench.delivery_stock}} | **{{ubench.delivery_prime}}** | {{ubench.delivery_paired_prime}} |
| Answers cut off by the thinking limit | {{ubench.n_prime}} | {{speed.cap_ubench_stock}} | **{{speed.cap_ubench_prime}}** | |

These rows cover {{auto.ubench_scope_prime}}. No benchmark question is quoted on this page.

### Can I just turn `reasoning_effort` down instead?

**For single questions, yes: lowering the effort works about as well as the Prime add-on.** For multi-step work and
long system prompts, no. This is the most important practical question on this page, so here is the full evidence.

**Single questions.** The original at effort {{tinker.effort_lo}}, and the original with the Prime add-on at
{{tinker.effort_hi}}, were each compared with the original at {{tinker.effort_hi}} on seven public benchmarks
({{tinker.items_stock}} questions). Measured on Thinking Machines' hosted service with the Prime add-on applied but
not merged. Both routes cut thinking sharply, and their accuracy differences are small and similar:

| | Original, effort {{tinker.effort_lo}} | Original + Prime add-on, effort {{tinker.effort_hi}} |
|---|---|---|
| Coding (HumanEval + MBPP, {{tinker.fam_coding_lo_n}} questions), difference in points | {{tinker.fam_coding_lo_diff}} | {{tinker.fam_coding_fix_diff}} |
| Bug fixing (HumanEvalFix + QuixBugs, {{tinker.fam_bugfix_lo_n}}) | {{tinker.fam_bugfix_lo_diff}} | {{tinker.fam_bugfix_fix_diff}} |
| Maths, number correct (GSM8K + MATH-500, {{tinker.fam_maths_number_lo_n}}) | {{tinker.fam_maths_number_lo_diff}} | {{tinker.fam_maths_number_fix_diff}} |
| Knowledge (MMLU, {{tinker.fam_knowledge_lo_n}}) | {{tinker.fam_knowledge_lo_diff}} | {{tinker.fam_knowledge_fix_diff}} |
| Mean thinking tokens per question (original at {{tinker.effort_hi}}: {{tinker.think_mean_stock}}) | {{tinker.think_mean_stock_lo}} | {{tinker.think_mean_fix}} |
| Answers cut off by the thinking limit (original at {{tinker.effort_hi}}: {{tinker.cap_stock}}) | {{tinker.cap_stock_lo}} | {{tinker.cap_fix}} |

**Multi-step debugging.** Here lowering the effort helps but does not cure the stalling. The same
{{effort_debug.n}} hard debugging tasks at three efforts, on the same hosted service:

| | Effort 0.3 | Effort 0.6 | Effort 0.9 |
|---|---|---|---|
| Original | {{effort_debug.stock_03}} | {{effort_debug.stock_06}} | {{effort_debug.stock_09}} |
| Original + Prime add-on | {{effort_debug.fix_03}} | {{effort_debug.fix_06}} | {{effort_debug.fix_09}} |

With the Prime add-on, the result is the same at every effort, so there is nothing to tune. This repo's weights at
effort 0.3 on a GPU server: {{effort_debug.prime_gpu_03}}.

**Long system prompts.** Every rule followed: the original {{sysprompt.kept_stock}} at effort 0.9 and
{{sysprompt.kept_stock_lo}} at 0.3; with the Prime add-on {{sysprompt.kept_fix}} at 0.9 and
{{sysprompt.kept_fix_lo}} at 0.3. Lowering the effort does not help the original here. Measured on Thinking
Machines' hosted service with the Prime add-on applied but not merged into the weights.

#### The same comparison on this repo's weights, on a GPU server

The release setup ({{speed.serving}}, {{speed.quant}}), each model at both efforts. Hard debugging is
{{prime_vs_stock.debug_n_tasks}} tasks; the benchmarks are the full sets.

| Model and effort | Hard debugging solved | Turns with no code | Tokens per task | HumanEvalFix | MBPP | MATH-500 | GSM8K, number correct | MMLU |
|---|---|---|---|---|---|---|---|---|
| **{{names.prime_name}}**, effort {{low_effort.effort_hi}} | {{gpu_low.debug_solved_prime_hi}} | {{gpu_low.debug_nocode_prime_hi}} | {{gpu_low.debug_tokens_prime_hi}} | {{gpu_low.humanevalfix_prime_hi}} | {{gpu_low.mbpp_prime_hi}} | {{gpu_low.math500_prime_hi}} | {{gpu_low.gsm8k_number_prime_hi}} | {{gpu_low.mmlu_prime_hi}} |
| {{names.prime_name}}, effort {{gpu_low.effort_lo}} | {{gpu_low.debug_solved_prime_lo}} | {{gpu_low.debug_nocode_prime_lo}} | {{gpu_low.debug_tokens_prime_lo}} | {{gpu_low.humanevalfix_prime_lo}} | {{gpu_low.mbpp_prime_lo}} | {{gpu_low.math500_prime_lo}} | {{gpu_low.gsm8k_number_prime_lo}} | {{gpu_low.mmlu_prime_lo}} |
| Original, effort {{low_effort.effort_hi}} | {{gpu_low.debug_solved_stock_hi}} | {{gpu_low.debug_nocode_stock_hi}} | {{gpu_low.debug_tokens_stock_hi}} | {{gpu_low.humanevalfix_stock_hi}} | {{gpu_low.mbpp_stock_hi}} | {{gpu_low.math500_stock_hi}} | {{gpu_low.gsm8k_number_stock_hi}} | {{gpu_low.mmlu_stock_hi}} |
| Original, effort {{gpu_low.effort_lo}} | {{gpu_low.debug_solved_stock_lo}} | {{gpu_low.debug_nocode_stock_lo}} | {{gpu_low.debug_tokens_stock_lo}} | {{gpu_low.humanevalfix_stock_lo}} | {{gpu_low.mbpp_stock_lo}} | {{gpu_low.math500_stock_lo}} | {{gpu_low.gsm8k_number_stock_lo}} | {{gpu_low.mmlu_stock_lo}} |

Strict-format maths was left out of the hosted-service table because it measures answer formatting: strict
difference {{tinker.fam_maths_strict_lo_diff}} at low effort, {{tinker.fam_maths_strict_fix_diff}} with the Prime
add-on.

**The honest conclusion.** If you only ask single questions, the original model at a lower effort is a fine
choice. {{names.prime_name}} is worth it when the model works in several steps (debugging loops, agents, tool use)
or follows a long system prompt, and when you would rather not tune the effort at all.

### Against Unbound and other releases

This table puts {{names.prime_name}} next to the original model, every Unbound version (see "Where this comes from"
above) and some uncensored Inkling 2 Mini releases by others. Each column says which release measured it. Blank
means that release did not measure it; nothing is estimated. "Useful" columns measure answers to harmful requests,
which the original and {{names.prime_name}} refuse; {{names.prime_name}} was not run on that test. **Drift** is how
far a model's word-by-word predictions on ordinary text have moved from the original's (KL divergence; smaller means
closer to the original).

{{auto.xver_table}}

![{{names.prime_name}}, the original model, the Unbound versions and other releases side by side](charts/xver_v3.png)

</details>

<details>
<summary><b>Do the smaller GGUF files keep the add-on's effect?</b>: mostly yes; hard debugging re-run on Q8_0, Q4_K_M and IQ2_M</summary>

Mostly yes: the smallest file, IQ2_M, still solves {{gguf.solve_prime_iq2m}} of the tasks but stalls on
{{gguf.nocode_prime_iq2m}} of turns; see the table. GGUF files are compressed ("quantised") copies of the weights;
Q8_0 is the largest and closest to the original weights, IQ2_M the smallest. Hard debugging re-run on the files with
llama.cpp ({{gguf.n_tasks}} tasks, same task set):

| File | Hard debugging solved | Turns with no code | Tokens per task | GSM8K, first 100, number correct | HumanEval, first 50 |
|---|---|---|---|---|---|
| {{names.prime_name}} Q8_0 | {{gguf.solve_prime_q8}} | {{gguf.nocode_prime_q8}} | {{gguf.tokens_prime_q8}} | {{gguf.gsm8k_number_prime_q8}} | {{gguf.humaneval_prime_q8}} |
| **{{names.prime_name}} Q4_K_M** | **{{gguf.solve_prime_q4km}}** | **{{gguf.nocode_prime_q4km}}** | **{{gguf.tokens_prime_q4km}}** | **{{gguf.gsm8k_number_prime_q4km}}** | **{{gguf.humaneval_prime_q4km}}** |
| {{names.prime_name}} IQ2_M | {{gguf.solve_prime_iq2m}} | {{gguf.nocode_prime_iq2m}} | {{gguf.tokens_prime_iq2m}} | {{gguf.gsm8k_number_prime_iq2m}} | {{gguf.humaneval_prime_iq2m}} |
| Original Q4_K_M, made the same way | {{gguf.solve_stock_q4km}} | {{gguf.nocode_stock_q4km}} | {{gguf.tokens_stock_q4km}} | {{gguf.gsm8k_number_stock_q4km}} | {{gguf.humaneval_stock_q4km}} |

How the files were built and checked is on the [GGUF card](https://huggingface.co/{{names.prime_gguf_repo}}).

</details>

<details>
<summary><b>Files in this repo</b>: the weights, configs and charts</summary>

| Path | What it is |
|---|---|
| `model-00001-of-00001.safetensors`, `mtp.safetensors`, configs, tokenizer, chat template | {{names.prime_name}}, ready to load. The sha256 of each file is in this repo's file list. |
| `charts/` | The charts on this page |

</details>

<details>
<summary><b>Limits</b>: where it can fall short, and what was not measured</summary>

- **No accuracy gain, and small coding losses not ruled out.** See "Did it lose anything?" under "Full results"
  above.
{{#if flags.capnote}}
- **When it does hit the thinking limit, the answer is usually wrong.** {{names.prime_name}} rarely runs into the
  limit, but when it does, the answer it is then forced to give tends to fail. The original's forced answers are
  usually still right. {{capnote.scope}}, {{capnote.n}} questions: the original hit the limit
  {{capnote.hits_stock}} times and {{capnote.right_stock}} of those answers passed; {{names.prime_name}} hit it
  {{capnote.hits_prime}} times and {{capnote.right_prime}} passed. On Thinking Machines' hosted service: the original
  {{capnote_tinker.stock}}; with the Prime add-on {{capnote_tinker.fix}}. The Prime add-on makes hitting the limit
  rare; it does not make a cut-off answer good.
{{/if}}
- **Shorter thinking can miss things.** Problems that genuinely need long reasoning may suffer.
- **Very close to the original, not identical.** Drift (how far its word-by-word predictions on ordinary text have
  moved from the original's): {{drift.kld_prime}}. Expect some wording differences.
- **Small debugging test.** The headline debugging result is {{prime_vs_stock.debug_n_tasks}} tasks. The direction
  is clear; the exact percentages are not precise.
- **One pass per model.** Apart from the seed re-runs, every number is a single run. Differences of one or two
  points on the small coding sets can be chance.
- **Not measured:** long documents, images and tool calling. Only English was tested.
- Everything the [original model's card](https://huggingface.co/{{names.stock_repo}}) says about capabilities,
  safety and limits still applies.

</details>

<details>
<summary><b>How it was made</b>: training method, data, checkpoint choice and merge</summary>

- **Starting point.** The original [{{names.stock_repo}}](https://huggingface.co/{{names.stock_repo}}), untouched.
- **The Prime add-on.** A LoRA adapter trained with **reinforcement learning**: the model attempted tasks and was
  rewarded for finishing them correctly without wasted thinking. Recipe: {{training.recipe}}. In plain terms: the
  learning rate sets how big each training update is; anchor answers are the original model's own answers (for each
  prompt, the best of eight attempts, "best-of-8"), trained on alongside so the model keeps its answer style.
- **Which checkpoint shipped.** We shipped {{names.adapter_checkpoint}}. The reason: {{checkpoint.choice}}.
- **Training data.** {{training.n_prompts}} harmless prompts written for this project ({{training.mix}}). No
  harmful requests and no benchmark questions. To keep answer quality from drifting, each step also trained on
  {{training.n_anchor}} of the original model's own answers.
- **Was it trained on the test questions?** No. To check, {{training.leak_check}} One overlap to disclose: 35 MBPP, 50 GSM8K
  and 5 MATH-500 questions also appear in the small in-house test sets used to pick the final checkpoint, so they
  were seen when choosing between checkpoints, never in training.
- **Merge.** The adapter was merged into the original weights ({{names.merge_method}}). The merge is repeatable: it
  produced a byte-identical file in {{files.prime_merges}} separate runs (sha256 `{{files.prime_sha}}`).

</details>

<details>
<summary><b>Reproduce the evaluation</b>: commands, settings, rebuilding the weights, per-question results</summary>

Evaluation code is in `scripts/evalsuite/` in the [GitHub repo](https://github.com/{{names.github_repo}}). Served
with {{reproduce.serving}} on {{reproduce.hardware}}. Settings: {{settings.text}}. The suite names and `--tag` values
in the commands are the harness's own labels; `bison` and `haiku` are the two grader models (Bison and Claude Haiku).

```bash
# generate (one tag per model; {{auto.ubench_limit_prime}})
python3 scripts/evalsuite/run_suite.py --base-url http://127.0.0.1:8000/v1 --model MODEL --tag {{names.prime_slug}} \
    --out-root results --suites agent_hard_noforce,ubench,main,hardmath,code,rp,pub_all \
    --effort {{settings.effort}} --temperature {{settings.temperature}} --top-p {{settings.top_p}} \
    --think-cap {{settings.think_cap}} --answer-cap {{settings.answer_cap}} --seed {{settings.seed}} --repeats 1 --allow-effort-unverified
# grade (paid judges): ubench with the grader model Bison; main, hardmath and rp with the two grader models (Bison and Claude Haiku)
python3 scripts/evalsuite/run_suite.py --base-url http://unused/v1 --model x --tag {{names.prime_slug}} --out-root results \
    --suites ubench --judge-only --judges bison --ubench-conditions command
python3 scripts/evalsuite/run_suite.py --base-url http://unused/v1 --model x --tag {{names.prime_slug}} --out-root results \
    --suites main,hardmath,rp --judge-only --judges bison,haiku
# intervals and paired tests, the original ("stock") as the reference
python3 scripts/evalsuite/compare.py stock {{names.prime_slug}} --ci --ref stock --margin 2 --out-root results --json ci.json \
    --suites pub_humaneval,pub_mbpp,pub_humanevalfix,pub_quixbugs,pub_gsm8k_full,pub_math500,pub_mmlu_2k,code,hardmath
```

To rebuild {{names.prime_name}} from the original model and the adapter: `merge_adapter.py --parts attn_experts
--rounding stochastic --seed 0 --device cuda` ({{files.merge_time}}; script and adapter in
[{{names.prime_lora_repo}}](https://huggingface.co/{{names.prime_lora_repo}}); adapter sha256 `{{files.lora_sha}}`).
The same adapter merged into Unbound v2.1 gives Unbound v3 (sha256 `{{files.unbound_sha}}`, byte-identical in
{{files.unbound_merges}} runs).

{{#if flags.per_item_published}}
Per-question scores for every benchmark (no model text for the sensitive-request benchmark):
{{reproduce.per_item_url}}
{{else}}
Per-question results are not published yet.
{{/if}}

</details>

## Intended use

For anyone who wants the original Inkling 2 Mini's behaviour, including its refusals, with less waiting and fewer
stalls: coding assistants, agents that work in several steps, chat with long system prompts, and everyday
questions. It is not an uncensored model; if that is what you need, see
[Inkling-2-Mini-Unbound](https://huggingface.co/{{names.unbound_repo}}). The base model's
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
  url    = {https://huggingface.co/{{names.prime_repo}}}
}
```
