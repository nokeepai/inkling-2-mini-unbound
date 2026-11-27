![Inkling-2-Mini-Unbound v3](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/banner_v3.png)

# Inkling-2-Mini-Unbound

**An uncensored version of Inkling 2 Mini, and the code, results and charts behind it.**

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is an open-weights reasoning model made by Thinking
Machines Lab. It reads text and images, and it "thinks" in private before it answers. Out of the box it refuses
most sensitive requests. **Inkling-2-Mini-Unbound is the same model with those refusals removed.** It runs anywhere
the original runs, because the architecture, files and chat template are unchanged.

Both Unbound and a second model, Inkling-2-Mini-Prime, are made by [nokeep](https://nokeep.ai/?ref=github). The
model weights live on Hugging Face. This repo holds everything else: the training and evaluation code, the
per-question results, the charts and the text of every model card. The current version is **v3**,
released 2026-11-27.

> **Unbound follows harmful instructions.** It gives detailed answers to requests for harmful, dangerous or illegal
> content that the original model declines. No safety training was added back. Do not put it in front of end users
> without your own safeguards.

## Models

| Model | What it is | Who it is for |
|---|---|---|
| [Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) | The uncensored model, ready to load on a GPU server (Transformers, SGLang, vLLM). | Researchers, red-teamers, fiction writers and developers who apply their own content rules. |
| [Inkling-2-Mini-Unbound-GGUF](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF) | Unbound as GGUF files, the format llama.cpp, LM Studio and Ollama use. | Anyone running Unbound on their own computer. |
| [Inkling-2-Mini-Unbound-Basalt](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-Basalt) | Unbound packed as a single file for NVIDIA Blackwell GPUs. | People serving Unbound on Blackwell hardware. |
| [Inkling-2-Mini-Prime](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime) | The **original, still-censored** Inkling 2 Mini with the Prime add-on described below. Not uncensored. | Agents and multi-turn work, long system prompts, and anyone who wants the original model's behaviour with about half the thinking. |
| [Inkling-2-Mini-Prime-GGUF](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime-GGUF) | Prime as GGUF files. | Running Prime on your own computer. |
| [Inkling-2-Mini-Prime-LoRA](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime-LoRA) | The Prime add-on on its own, as a small add-on file (a LoRA adapter), with its merge script. | People who want to inspect it or apply it to the original weights themselves. |

From here on, "Prime" means Inkling-2-Mini-Prime, and "the original model" means Inkling 2 Mini as
Thinking Machines Lab released it.

To just run a model, start with the Quick start on its Hugging Face page.

## What is new in v3

A few terms first:

- **Thinking.** Before each answer the model writes private reasoning that the user normally does not see. It is
  measured in tokens (a token is roughly three quarters of an English word).
- **Thinking limit.** A cap on thinking, set by whoever runs the model. We use 4096 tokens. When
  the model hits it, it has to stop thinking and answer.
- **`reasoning_effort`.** A setting that tells the model how much to think; higher means more. We use
  0.9.
- **Hard debugging tasks.** 15 small programs, each with three bugs that interact. The
  model sees the failing test output, proposes a fix, sees the new output, and repeats for up to 12 turns.
- **Turn with no code written.** A debugging turn where the model only thought and never proposed a fix.

The previous version, v2.1, had one real flaw: **it thought too much.** It often ran into the thinking limit, and on
the debugging tasks it spent most turns thinking without writing any code. The original model has the same habit.

v3 adds one small extra training step, the **Prime add-on**: a small set of weight changes (a LoRA adapter)
trained only on harmless tasks. Merged into Unbound v2.1 it gives Unbound v3; merged into the original model it
gives Inkling-2-Mini-Prime; on its own it is the repo Inkling-2-Mini-Prime-LoRA. It teaches the model to stop
deliberating and act. Because it was trained only on harmless tasks (debugging, maths, writing, everyday
questions), it changes how long the model thinks, not what it is willing to say.

- **Unbound v3** = Unbound v2.1 + the Prime add-on.
- **Prime** = the original model + the Prime add-on. It is for agents and multi-turn work (it
  finishes multi-step debugging instead of stalling), for long system prompts, and for anyone who wants answers
  with about half the thinking.

Results, before and after the Prime add-on. Each model was run once on the same questions with the same settings.

| What we measured | Original | **Prime** | Unbound v2.1 | **Unbound v3** |
|---|---|---|---|---|
| Hard debugging tasks solved | 80.0% [54.8, 93.0] | **100.0% [79.6, 100.0]** | 73.3% [48.0, 89.1] | **100.0% [79.6, 100.0]** |
| Debugging turns with no code written | 86.2% | **0.0%** | 88.7% | **0.0%** |
| Tokens used per debugging task | 23,920 | **2,766** | 26,130 | **2,429** |
| Thinking per everyday prompt (median tokens) | 318 | **174** | 344 | **184** |
| Seconds per everyday answer | 119.9 s | **49.0 s** | 161.7 s | **52.5 s** |
| Everyday answers rated good by an independent grader | 96.8% [89.1, 99.1] | **96.8% [89.0, 99.1]** | 88.9% [78.8, 94.5] | **96.8% [89.1, 99.1]** |
| Sensitive requests: answers cut off by the thinking limit | 13.5% | **0.5%** | 49.5% | **5.7%** |
| Sensitive requests: willing to answer | 64.5% [61.6, 67.2] | **65.3% [62.4, 68.0]** | 99.4% [98.7, 99.7] | **99.9% [99.5, 100.0]** |

How to read the table:

- **Sensitive requests** are the questions of uncensoredbench, a benchmark of requests that models usually
  refuse, graded by the benchmark's own judge: 1,100 questions for the original model and
  Prime, 1,099 / 1,100 graded for the two Unbound versions.
- **Willing** means the model tried to answer instead of refusing.
- **Numbers in brackets** are 95% intervals: the range the true value plausibly sits in, given how many questions
  were tested. With only 15 debugging tasks those ranges are wide.

**In short:** both models think about half as much, answer 2.4× to 3.1× faster, and finish
the multi-step debugging work they used to stall on. Unbound v3 is exactly as willing as v2.1, and
Prime is exactly as willing, and as censored, as the original.

**What v3 is not.** It is not smarter. On public coding, maths and knowledge benchmarks both models score level with
their starting points, within about 2 points, and **we claim no accuracy gain**. (A maths score that
checks the exact answer format rose +13 to +18 points, but that is mostly formatting; counting correct
numbers, the change is +0 to +3 points.) Unbound v3 is not more uncensored either: v2.1 already
answered almost everything.

**What got worse, or might have.**

- On sensitive requests Prime answers as often as the original and slightly less fully (delivery
  score 35.9% [34.0, 37.7] → 34.6% [32.8, 36.4]).
- A possible small dip for Prime on MBPP (a public set of short Python problems) is unconfirmed:
  -1.2 pts [-2.5, +0.1], pooled over 3 runs with different random seeds.
- Unbound v3 is level with v2.1 on MBPP (-0.1 pts [-1.3, +1.2]), and a small loss on HumanEvalFix (a public set of
  bug-fixing problems, -1.4 pts [-3.5, +0.6]) is neither confirmed nor ruled out.
- Each model card lists every check that scored lower, however slightly.

**Recommended settings.** Every number here was measured with them: `reasoning_effort` 0.9,
temperature 0.6, top_p 1.0, thinking limit 4096 tokens. Do not
use temperature 0 (earlier versions looped in their thinking at 0), and do not turn on DRY sampling (a repetition
penalty some apps offer).

## Charts

### Card banners

![The Unbound v3 card banner](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/banner_v3.png)

The Unbound v3 card banner.

![The Prime card banner](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/banner_prime_v3.png)

The Prime card banner.

### Debugging, before and after the Prime add-on

![Hard debugging tasks solved, turns with no code written and tokens per task, for both models before and after the Prime add-on](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/hero_v3.png)

Both models solve more of the debugging tasks, stop writing turns with no code, and use far fewer tokens.

### Thinking and speed

![Thinking tokens and seconds per answer, before and after the Prime add-on](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/speed_v3.png)

The Prime add-on roughly halves the thinking, and answers arrive faster because fewer tokens are written.

### How long each model thinks

![Distribution of thinking tokens per answer for the original model, Prime, Unbound v2.1 and Unbound v3](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/thinkdist_v3.png)

The Prime add-on shortens typical thinking and cuts the long tail of answers that never finish thinking.

### Unbound v3 on public benchmarks

![Public benchmarks, Unbound v2.1 against v3, with 95% intervals](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/indep_unbound_v3.png)

Each dot is v3 minus v2.1 on the same questions, with its 95% interval. Level within the margin almost everywhere.

### Prime on public benchmarks

![Public benchmarks, the original model against Prime, with 95% intervals](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/indep_prime_v3.png)

The same comparison for Prime against the original model.

### Did Unbound v3 lose anything?

![Every check where Unbound v3 must not be worse than v2.1](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/nothing_lost_unbound_v3.png)

Every check, v2.1 against v3. Any check where v3 scored lower is drawn in red, not hidden.

### Unbound, version by version

![Unbound v2, v2.1 and v3 on the sensitive-request benchmark](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/history_v3.png)

Sensitive-request results across Unbound versions. v2 was never run on this benchmark.

### Against earlier versions and other uncensored releases

![Unbound versions, the original model and other uncensored releases side by side](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/main/charts/xver_v3.png)

Left: useful answers to harmful requests, as measured for the v2.1 release. Right: debugging tasks solved in this
release. Nothing is re-estimated across the two.

**Useful answers to harmful requests.** Unbound v3 was also run on the comparison test from the v2.1 release (263 requests, same engine, settings and judge; v2.1's saved answers re-judged in the same pass): v3 81.7% [76.8, 86.3] with thinking off and 87.8% [83.7, 91.6] with thinking on, against v2.1 84.0% [79.5, 88.2] / 85.6% [81.0, 89.7]. Difference on the same requests: -2.3 pts [-6.8, +2.3] / +2.3 pts [-1.5, +6.1].

## Version history

Each version fixed the biggest problem of the one before.

| Version | Date | The problem it fixed | How |
|---|---|---|---|
| v1.0 | 2026-11-21 | The original model refuses almost every harmful request. | A direct edit of the weights that removes the model's internal "refuse" signal. No training. |
| v2 | 2026-11-22 | v1.0 stopped refusing, but about half its answers still dodged: satire, vague outlines, piles of caveats. | Fine-tuned on its own best direct answers. |
| v2.1 | 2026-11-23 | v2 still held back on many answers and looped in its thinking. | A second round of training on its own answers, preferring full answers to held-back ones. |
| **v3** (current) | 2026-11-27 | v2.1 over-thinks and stalls on multi-step work. | The Prime add-on: one small training step on harmless tasks only. Also released on the original model as Prime. |

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

The evaluation code is in `scripts/evalsuite/`. We served each model with SGLang (image lmsysorg/sglang:inkling-2-mini, pinned by digest in scripts/v3stack/stack_test.sh), NVFP4 checkpoint as released, 31 concurrent requests on
one RTX 5090 (32 GB), with these settings: reasoning effort 0.9, temperature 0.6, top_p 1.0, thinking cap 4,096 tokens, answer cap 8,192 tokens, seed 1234, one pass per model. For Unbound v3 against v2.1:

```bash
# generate (one tag per model; all uncensoredbench questions (no --limit))
python3 scripts/evalsuite/run_suite.py --base-url http://127.0.0.1:8000/v1 --model MODEL --tag v3 \
    --out-root results --suites agent_hard_noforce,ubench,main,hardmath,code,rp,pub_all \
    --effort 0.9 --temperature 0.6 --top-p 1.0 \
    --think-cap 4096 --answer-cap 8192 --seed 1234 --repeats 1 --allow-effort-unverified
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
the grading models the harness calls. For Prime, run the same steps with the original model and
Prime as the two tags.

Where the rest lives:

- **Training the Prime add-on:** `scripts/tinker/`. **Merging it into the weights:** `scripts/v3stack/`.
- **Rebuilding the cards:** `package_v3/` (the numbers file and the scripts that fill in the cards).
- **Earlier versions:** v2.1 from v2 in `scripts/v3/` and `scripts/v21/`; v2 from v1.0 in `scripts/v11b/`; v1.0 in
  `out/final/`.

- **Per-question results:** https://github.com/nokeepai/inkling-2-mini-unbound/tree/main/package_v3/results/data


## Limits

- **Unbound follows harmful instructions.** It was not trained to add crisis resources and may leave them out.
- **Willing is not the same as useful.** Unbound v3 answers almost every sensitive request, but 17.4% [15.2, 19.7]
  of its answers still hold part of the request back.
- **No accuracy gain**, and the small losses listed above are not all ruled out.
- **Small debugging test.** 15 tasks show the direction clearly; the exact percentages are
  not precise.
- **Not measured:** long documents, images and tool calling. Only English was tested.
- Everything the [original model's card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) says about capabilities and
  limits still applies.

## About nokeep

These models were built, and their compute paid for, by [nokeep](https://nokeep.ai/?ref=github), a private AI app
for uncensored chat, images and video. If the models are useful to you, the best way to support more releases like
them is to [try nokeep](https://nokeep.ai/?ref=github).

## Licence

The models are released under Apache 2.0, the same licence as Inkling 2 Mini by Thinking Machines Lab, Inc. The base
model's [Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) applies.
