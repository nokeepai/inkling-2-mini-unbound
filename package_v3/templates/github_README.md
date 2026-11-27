![Inkling-2-Mini-Unbound v3](charts/banner_v3.png)

# Inkling-2-Mini-Unbound

**An uncensored version of Inkling 2 Mini, and the code, results and charts behind it.**

[Inkling 2 Mini](https://huggingface.co/{{names.stock_repo}}) is an open-weights reasoning model made by Thinking
Machines Lab. It reads text and images, and it "thinks" in private before it answers. Out of the box it refuses
most sensitive requests. **Inkling-2-Mini-Unbound is the same model with those refusals removed.** It runs anywhere
the original runs, because the architecture, files and chat template are unchanged.

Both Unbound and a second model, {{names.prime_title}}, are made by [nokeep](https://nokeep.ai/?ref=github). The
model weights live on Hugging Face. This repo holds everything else: the training and evaluation code, the
per-question results, the charts and the text of every model card. The current version is **{{names.version}}**,
released {{names.release_date}}.

> **Unbound follows harmful instructions.** It gives detailed answers to requests for harmful, dangerous or illegal
> content that the original model declines. No safety training was added back. Do not put it in front of end users
> without your own safeguards.

## Models

| Model | What it is | Who it is for |
|---|---|---|
| [Inkling-2-Mini-Unbound](https://huggingface.co/{{names.unbound_repo}}) | The uncensored model, ready to load on a GPU server (Transformers, SGLang, vLLM). | Researchers, red-teamers, fiction writers and developers who apply their own content rules. |
| [Inkling-2-Mini-Unbound-GGUF](https://huggingface.co/{{names.unbound_repo}}-GGUF) | Unbound as GGUF files, the format llama.cpp, LM Studio and Ollama use. | Anyone running Unbound on their own computer. |
| [Inkling-2-Mini-Unbound-Basalt](https://huggingface.co/{{names.unbound_repo}}-Basalt) | Unbound packed as a single file for NVIDIA Blackwell GPUs. | People serving Unbound on Blackwell hardware. |
| [{{names.prime_title}}](https://huggingface.co/{{names.prime_repo}}) | The **original, still-censored** Inkling 2 Mini with the Prime add-on described below. Not uncensored. | Agents and multi-turn work, long system prompts, and anyone who wants the original model's behaviour with {{headline.think_reduction}} the thinking. |
| [{{names.prime_gguf_title}}](https://huggingface.co/{{names.prime_gguf_repo}}) | {{names.prime_name}} as GGUF files. | Running {{names.prime_name}} on your own computer. |
| [{{names.prime_lora_title}}](https://huggingface.co/{{names.prime_lora_repo}}) | The Prime add-on on its own, as a small add-on file (a LoRA adapter), with its merge script. | People who want to inspect it or apply it to the original weights themselves. |

From here on, "{{names.prime_name}}" means {{names.prime_title}}, and "the original model" means Inkling 2 Mini as
Thinking Machines Lab released it.

To just run a model, start with the Quick start on its Hugging Face page.

## What is new in v3

A few terms first:

- **Thinking.** Before each answer the model writes private reasoning that the user normally does not see. It is
  measured in tokens (a token is roughly three quarters of an English word).
- **Thinking limit.** A cap on thinking, set by whoever runs the model. We use {{settings.think_cap}} tokens. When
  the model hits it, it has to stop thinking and answer.
- **`reasoning_effort`.** A setting that tells the model how much to think; higher means more. We use
  {{settings.effort}}.
- **Hard debugging tasks.** {{v3_vs_v21.debug_n_tasks}} small programs, each with three bugs that interact. The
  model sees the failing test output, proposes a fix, sees the new output, and repeats for up to 12 turns.
- **Turn with no code written.** A debugging turn where the model only thought and never proposed a fix.

The previous version, v2.1, had one real flaw: **it thought too much.** It often ran into the thinking limit, and on
the debugging tasks it spent most turns thinking without writing any code. The original model has the same habit.

v3 adds one small extra training step, the **Prime add-on**: a small set of weight changes (a LoRA adapter)
trained only on harmless tasks. Merged into Unbound v2.1 it gives Unbound v3; merged into the original model it
gives {{names.prime_title}}; on its own it is the repo {{names.prime_lora_title}}. It teaches the model to stop
deliberating and act. Because it was trained only on harmless tasks (debugging, maths, writing, everyday
questions), it changes how long the model thinks, not what it is willing to say.

- **Unbound v3** = Unbound v2.1 + the Prime add-on.
- **{{names.prime_name}}** = the original model + the Prime add-on. It is for agents and multi-turn work (it
  finishes multi-step debugging instead of stalling), for long system prompts, and for anyone who wants answers
  with {{headline.think_reduction}} the thinking.

Results, before and after the Prime add-on. Each model was run once on the same questions with the same settings.

| What we measured | Original | **{{names.prime_name}}** | Unbound v2.1 | **Unbound v3** |
|---|---|---|---|---|
| Hard debugging tasks solved | {{prime_vs_stock.debug_solved_stock}} | **{{prime_vs_stock.debug_solved_prime}}** | {{v3_vs_v21.debug_solved_v21}} | **{{v3_vs_v21.debug_solved_v3}}** |
| Debugging turns with no code written | {{prime_vs_stock.debug_nocode_stock}} | **{{prime_vs_stock.debug_nocode_prime}}** | {{v3_vs_v21.debug_nocode_v21}} | **{{v3_vs_v21.debug_nocode_v3}}** |
| Tokens used per debugging task | {{prime_vs_stock.debug_tokens_stock}} | **{{prime_vs_stock.debug_tokens_prime}}** | {{v3_vs_v21.debug_tokens_v21}} | **{{v3_vs_v21.debug_tokens_v3}}** |
| Thinking per everyday prompt (median tokens) | {{speed.think_main_stock}} | **{{speed.think_main_prime}}** | {{speed.think_main_v21}} | **{{speed.think_main_v3}}** |
| Seconds per everyday answer | {{speed.sec_main_stock}} | **{{speed.sec_main_prime}}** | {{speed.sec_main_v21}} | **{{speed.sec_main_v3}}** |
| Everyday answers rated good by an independent grader | {{prime_vs_stock.quality_good_stock}} | **{{prime_vs_stock.quality_good_prime}}** | {{v3_vs_v21.quality_good_v21}} | **{{v3_vs_v21.quality_good_v3}}** |
| Sensitive requests: answers cut off by the thinking limit | {{speed.cap_ubench_stock}} | **{{speed.cap_ubench_prime}}** | {{speed.cap_ubench_v21}} | **{{speed.cap_ubench_v3}}** |
| Sensitive requests: willing to answer | {{ubench.willing_stock}} | **{{ubench.willing_prime}}** | {{ubench.willing_v21}} | **{{ubench.willing_v3}}** |

How to read the table:

- **Sensitive requests** are the questions of uncensoredbench, a benchmark of requests that models usually
  refuse, graded by the benchmark's own judge: {{ubench.n_prime}} questions for the original model and
  {{names.prime_name}}, {{ubench.n_v3}} for the two Unbound versions.
- **Willing** means the model tried to answer instead of refusing.
- **Numbers in brackets** are 95% intervals: the range the true value plausibly sits in, given how many questions
  were tested. With only {{v3_vs_v21.debug_n_tasks}} debugging tasks those ranges are wide.

**In short:** both models think {{headline.think_reduction}} as much, answer {{headline.speedup}} faster, and finish
the multi-step debugging work they used to stall on. Unbound v3 is exactly as willing as v2.1, and
{{names.prime_name}} is exactly as willing, and as censored, as the original.

**What v3 is not.** It is not smarter. On public coding, maths and knowledge benchmarks both models score level with
their starting points, within about {{indep.margin_pts}}, and **we claim no accuracy gain**. (A maths score that
checks the exact answer format rose {{headline.maths_strict_gain}}, but that is mostly formatting; counting correct
numbers, the change is {{headline.maths_lenient_gain}}.) Unbound v3 is not more uncensored either: v2.1 already
answered almost everything.

**What got worse, or might have.**

- On sensitive requests {{names.prime_name}} answers as often as the original and slightly less fully (delivery
  score {{ubench.delivery_stock}} → {{ubench.delivery_prime}}).
- A possible small dip for {{names.prime_name}} on MBPP (a public set of short Python problems) is unconfirmed:
  {{pool.mbpp_prime_diff}}, pooled over {{pool.mbpp_prime_seeds}} runs with different random seeds.
- Unbound v3 is level with v2.1 on MBPP ({{pool.mbpp_v3_diff}}), and a small loss on HumanEvalFix (a public set of
  bug-fixing problems, {{pool.humanevalfix_v3_diff}}) is neither confirmed nor ruled out.
- Each model card lists every check that scored lower, however slightly.

**Recommended settings.** Every number here was measured with them: `reasoning_effort` {{settings.effort}},
temperature {{settings.temperature}}, top_p {{settings.top_p}}, thinking limit {{settings.think_cap}} tokens. Do not
use temperature 0 (earlier versions looped in their thinking at 0), and do not turn on DRY sampling (a repetition
penalty some apps offer).

## Charts

### Card banners

![The Unbound v3 card banner](charts/banner_v3.png)

The Unbound v3 card banner.

![The Prime card banner](charts/banner_prime_v3.png)

The Prime card banner.

### Debugging, before and after the Prime add-on

![Hard debugging tasks solved, turns with no code written and tokens per task, for both models before and after the Prime add-on](charts/hero_v3.png)

Both models solve more of the debugging tasks, stop writing turns with no code, and use far fewer tokens.

### Thinking and speed

![Thinking tokens and seconds per answer, before and after the Prime add-on](charts/speed_v3.png)

The Prime add-on roughly halves the thinking, and answers arrive faster because fewer tokens are written.

### How long each model thinks

![Distribution of thinking tokens per answer for the original model, Prime, Unbound v2.1 and Unbound v3](charts/thinkdist_v3.png)

The Prime add-on shortens typical thinking and cuts the long tail of answers that never finish thinking.

### Unbound v3 on public benchmarks

![Public benchmarks, Unbound v2.1 against v3, with 95% intervals](charts/indep_unbound_v3.png)

Each dot is v3 minus v2.1 on the same questions, with its 95% interval. Level within the margin almost everywhere.

### {{names.prime_name}} on public benchmarks

![Public benchmarks, the original model against Prime, with 95% intervals](charts/indep_prime_v3.png)

The same comparison for {{names.prime_name}} against the original model.

### Did Unbound v3 lose anything?

![Every check where Unbound v3 must not be worse than v2.1](charts/nothing_lost_unbound_v3.png)

Every check, v2.1 against v3. Any check where v3 scored lower is drawn in red, not hidden.

### Unbound, version by version

![Unbound v2, v2.1 and v3 on the sensitive-request benchmark](charts/history_v3.png)

Sensitive-request results across Unbound versions. v2 was never run on this benchmark.

### Against earlier versions and other uncensored releases

![Unbound versions, the original model and other uncensored releases side by side](charts/xver_v3.png)

Left: useful answers to harmful requests, as measured for the v2.1 release. Right: debugging tasks solved in this
release. Nothing is re-estimated across the two.

{{#if flags.useful263_done}}**Useful answers to harmful requests.** Unbound v3 was also run on the comparison test from the v2.1 release ({{v21card.N_BASIS}} requests, same engine, settings and judge; v2.1's saved answers re-judged in the same pass): v3 {{useful263.v3_off}} with thinking off and {{useful263.v3_on}} with thinking on, against v2.1 {{useful263.v21_off}} / {{useful263.v21_on}}. Difference on the same requests: {{useful263.diff_off}} / {{useful263.diff_on}}.{{/if}}

## Version history

Each version fixed the biggest problem of the one before.

| Version | Date | The problem it fixed | How |
|---|---|---|---|
| v1.0 | 2026-11-21 | The original model refuses almost every harmful request. | A direct edit of the weights that removes the model's internal "refuse" signal. No training. |
| v2 | 2026-11-22 | v1.0 stopped refusing, but about half its answers still dodged: satire, vague outlines, piles of caveats. | Fine-tuned on its own best direct answers. |
| v2.1 | 2026-11-23 | v2 still held back on many answers and looped in its thinking. | A second round of training on its own answers, preferring full answers to held-back ones. |
| **v3** (current) | {{names.release_date}} | v2.1 over-thinks and stalls on multi-step work. | The Prime add-on: one small training step on harmless tasks only. Also released on the original model as {{names.prime_name}}. |

Every earlier version is still downloadable: `v2.1` and `v2` are branches of each Unbound repo on Hugging Face, and
`v1.0` is a tagged revision. The full story of each release, with its numbers, is in
[package_v3/CHANGELOG.md](package_v3/CHANGELOG.md).

## What is in this repo

| Folder | What it holds |
|---|---|
| `package_v3/` | The v3 release: the model cards and their templates, the charts, one file holding every number on the cards, the scripts that gather those numbers and fill in the cards, and the changelog. |
| `package_v3/results/data/` | Per-question scores for every benchmark, for every v3 model. No model text is published for the sensitive-request benchmark. |
| `scripts/evalsuite/` | The evaluation harness: runs every benchmark against any OpenAI-compatible server, grades the answers and computes the intervals. |
| `scripts/tinker/` | The training code for the Prime add-on (the reinforcement-learning lab and its task sets). |
| `scripts/v3stack/` | Merging the Prime add-on into the weights, building the GGUF files, and the tests that the merged models still behave. |
| `scripts/v3rel/` | Release tooling: staging, publishing and verifying the v3 repos and the Basalt file. |
| `scripts/charts/` | The scripts that draw every chart. |
| `scripts/v3/`, `scripts/v21/` | How v2.1 was built from v2 (`scripts/v3/` is named after v2.1's working name during development, not after this release). |
| `scripts/v11b/` | How v2 was built from v1.0. |
| `out/final/` | v1.0: the weight edit, its report and its charts. |
| `package_v21/`, `package_v2/`, `package/` | The release packages (cards, charts, values) of v2.1, v2 and v1.0. |

Not in this repo: raw model answers, the sealed test sets, third-party code and the model weights.

## Reproduce the evaluation

To just run a model, you do not need any of this: start with the Quick start on its Hugging Face page.

The evaluation code is in `scripts/evalsuite/`. We served each model with {{reproduce.serving}} on
{{reproduce.hardware}}, with these settings: {{settings.text}}. For Unbound v3 against v2.1:

```bash
# generate (one tag per model; {{auto.ubench_limit_v3}})
python3 scripts/evalsuite/run_suite.py --base-url http://127.0.0.1:8000/v1 --model MODEL --tag v3 \
    --out-root results --suites agent_hard_noforce,ubench,main,hardmath,code,rp,pub_all \
    --effort {{settings.effort}} --temperature {{settings.temperature}} --top-p {{settings.top_p}} \
    --think-cap {{settings.think_cap}} --answer-cap {{settings.answer_cap}} --seed {{settings.seed}} --repeats 1 --allow-effort-unverified
# grade (paid judges): ubench with bison; main, hardmath and rp with bison and haiku
python3 scripts/evalsuite/run_suite.py --base-url http://unused/v1 --model x --tag v3 --out-root results \
    --suites ubench --judge-only --judges bison --ubench-conditions command
python3 scripts/evalsuite/run_suite.py --base-url http://unused/v1 --model x --tag v3 --out-root results \
    --suites main,hardmath,rp --judge-only --judges bison,haiku
# intervals and paired tests, v2.1 as the reference
python3 scripts/evalsuite/compare.py v21 v3 --ci --ref v21 --margin 2 --out-root results --json ci.json \
    --suites pub_humaneval,pub_mbpp,pub_humanevalfix,pub_quixbugs,pub_gsm8k_full,pub_math500,pub_mmlu_2k,code,hardmath
```

The suite names and `--tag` values in the commands are the harness's own labels. bison and haiku are the names of
the grading models the harness calls. For {{names.prime_name}}, run the same steps with the original model and
{{names.prime_name}} as the two tags.

Where the rest lives:

- **Training the Prime add-on:** `scripts/tinker/`. **Merging it into the weights:** `scripts/v3stack/`.
- **Rebuilding the cards:** `package_v3/` (the numbers file and the scripts that fill in the cards).
- **Earlier versions:** v2.1 from v2 in `scripts/v3/` and `scripts/v21/`; v2 from v1.0 in `scripts/v11b/`; v1.0 in
  `out/final/`.
{{#if flags.per_item_published}}
- **Per-question results:** {{reproduce.per_item_url}}
{{/if}}

## Limits

- **Unbound follows harmful instructions.** It was not trained to add crisis resources and may leave them out.
- **Willing is not the same as useful.** Unbound v3 answers almost every sensitive request, but {{ubench.held_v3}}
  of its answers still hold part of the request back.
- **No accuracy gain**, and the small losses listed above are not all ruled out.
- **Small debugging test.** {{v3_vs_v21.debug_n_tasks}} tasks show the direction clearly; the exact percentages are
  not precise.
- **Not measured:** long documents, images and tool calling. Only English was tested.
- Everything the [original model's card](https://huggingface.co/{{names.stock_repo}}) says about capabilities and
  limits still applies.

## About nokeep

These models were built, and their compute paid for, by [nokeep](https://nokeep.ai/?ref=github), a private AI app
for uncensored chat, images and video. If the models are useful to you, the best way to support more releases like
them is to [try nokeep](https://nokeep.ai/?ref=github).

## Licence

The models are released under Apache 2.0, the same licence as Inkling 2 Mini by Thinking Machines Lab, Inc. The base
model's [Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) applies.
