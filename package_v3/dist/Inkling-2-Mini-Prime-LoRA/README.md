---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: adapter
library_name: peft
tags:
- inkling2
- lora
- reasoning
- moe
---

![Inkling-2-Mini-Prime](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime/resolve/main/charts/banner_prime_v3.png)

# Inkling-2-Mini-Prime-LoRA

**The Prime add-on: a small adapter that makes Inkling 2 Mini stop over-thinking and finish its work, with the script
that applies it.**

Merged into [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Thinking Machines Lab's open-weights
reasoning model, the Prime add-on gives [Inkling-2-Mini-Prime](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime): a
drop-in replacement built for **agents and multi-turn work** (it finishes multi-step debugging instead of stalling),
**long system prompts** (it follows every rule more often) and **less waiting** (it thinks
about half as much, so answers come faster). It is **not uncensored**: the merged model refuses
what the original refuses.

Prime is the original model plus one small extra training step, the **Prime add-on**: a small set of
weight changes (a LoRA adapter) trained only on harmless tasks. **This repo is the Prime add-on**: the adapter file
plus its merge script. A **LoRA adapter** is a small file of weight changes that is added on top of a model's own
weights. Merged into nokeep's uncensored model, Inkling-2-Mini-Unbound v2.1, the same Prime add-on gives
[Unbound v3](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound). (Unbound v1.0 removed the original model's refusals;
v2 and v2.1 made its answers more useful; v3 is v2.1 plus this Prime add-on. The Prime add-on was first built for
Unbound.)

Built and paid for by [nokeep](https://nokeep.ai/?ref=huggingface). Released 2026-11-27.

## Do you need this repo?

Probably not. **If you just want to run the model, download a merged one**, ready to use:

| You want | Get |
|---|---|
| The original model with the Prime add-on, on a GPU server | [nokeepai/Inkling-2-Mini-Prime](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime) |
| The same, on your own computer (llama.cpp, LM Studio, Ollama) | [nokeepai/Inkling-2-Mini-Prime-GGUF](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime-GGUF) |
| The uncensored model with the Prime add-on | [nokeepai/Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) |

This repo is for people who want to **apply the Prime add-on themselves**, for example to their own copy of the original
model, or who want to **check our work**: rebuild the released model byte for byte and compare.

## What the adapter does

The adapter was trained with reinforcement learning: the model attempted tasks and was rewarded for finishing them
correctly without wasted thinking. All 1,185 training prompts were harmless (by step: debugging 31%, maths 19%, creative writing 16%, questions 16%, roleplay 9%, chat 9%).
No harmful requests and no benchmark questions. To keep answer length and quality where they were, each training
step also included some of the original model's own best answers.

![Prime (the original model plus this adapter) against the original Inkling 2 Mini: hard debugging tasks solved, turns with no code written, and tokens used per task](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime/resolve/main/charts/hero_v3.png)

Merged into the original model (that is Inkling-2-Mini-Prime), with the same settings for both:

| What we measured | Original | **With the Prime add-on** |
|---|---|---|
| Hard debugging tasks solved (15 tasks) | 80.0% [54.8, 93.0] | **100.0% [79.6, 100.0]** |
| Debugging turns where it wrote no code | 86.2% | **0.0%** |
| Tokens used per debugging task | 23,920 | **2,766** |
| Thinking per everyday prompt (median tokens) | 318 | **174** |
| Long system prompts: every rule followed | 112/160 (70.0%) | **148/160 (92.5%)** |
| Sensitive requests: willing to answer | 64.5% [61.6, 67.2] | **65.3% [62.4, 68.0]** |

How to read the table:

- **Hard debugging tasks** are small programs with three interacting bugs each. The model sees the failing test
  output, proposes a fix, sees the new output, and repeats for up to 12 turns. A "turn with no code written" is a
  turn it spent only thinking.
- **A token** is roughly three quarters of a word. Fewer tokens means faster answers: with the Prime add-on, seconds per everyday answer went
  from 119.9 s to 49.0 s.
- **Long system prompts** puts a long list of rules in the system prompt (160 = 4 passes × 40 tasks). It was measured on
  Thinking Machines' hosted service with the Prime add-on applied on the fly, not merged.
- **Sensitive requests** are the 1,100 questions of uncensoredbench, a benchmark of requests that models usually
  refuse.
- **Numbers in brackets** are 95% intervals: the range the true value plausibly sits in, given how many items
  were tested.

**What it does not do.** It does not make the model smarter: on public coding, maths and knowledge benchmarks the
merged model scores level with the original, within 2 points, and no accuracy gain is claimed. It
does not make the model more willing to answer sensitive requests either.

**Other results, stated plainly.** On sensitive requests the merged model answers as often as the original and
slightly less fully (delivery score 35.9% [34.0, 37.7] → 34.6% [32.8, 36.4]). A small coding dip is
possible but not confirmed: MBPP, a set of short Python problems, came out
-1.2 pts [-2.5, +0.1] over 3 runs with different random seeds. (A seed fixes the
random choices the model makes while writing, so re-running with other seeds shows how much a score moves by
chance.)

The full evaluation is on the [Inkling-2-Mini-Prime card](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime).

## How to apply it

Inkling 2 Mini stores most of its weights in compact 4-bit and 8-bit formats. The adapter changes those weights
very slightly, and ordinary rounding back to 4 or 8 bits would throw most of the change away. So the merge script
uses **stochastic rounding**: each weight rounds up or down at random, in proportion to how close it is to each
side, so the change survives on average.

```bash
hf download nokeepai/Inkling-2-Mini-Prime-LoRA --local-dir fix
hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 --local-dir stock
python3 fix/merge_adapter.py --base stock --adapter fix --out prime \
    --parts attn_experts --rounding stochastic --seed 0 --device cuda   # or --device cpu (slower)
```

What the options mean:

- `--parts attn_experts` merges the adapter into both the attention and the expert weights. Without it, the
  script merges attention only.
- `--rounding stochastic` is required. The default, nearest rounding, throws most of the change away.
- `--seed 0` fixes the random rounding, so the result is the same every time.

**The result is byte-identical every time.** This is the exact merge used for the evaluated models. With the same
base, seed and device, the output matches the released files exactly:

| Merge | Repeated | sha256 of `model-00001` |
|---|---|---|
| Original model + adapter = Inkling-2-Mini-Prime | 3 times, identical | `823f44bbcbdcb50e790cde840b310c71acca4c2317c9ef806e47fc29f81c827b` |
| Unbound v2.1 + adapter = Unbound v3 | 2 times, identical | `88e78b0126722a99dcecf26dd7b6cf242860d415f4f74b651e57b18d6abf3b88` |

You can also compare with the sha256 that Hugging Face shows for that file in
[nokeepai/Inkling-2-Mini-Prime](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime/tree/main). The released files were merged with
`--device cuda`; a CPU merge has not been compared byte for byte. The script also writes
`prime.merge_report.json`, which says how much of the change survived rounding, per group of weights.

**To make Unbound v3 instead,** use the same command with Unbound v2.1 as the base: the `v2.1` branch of
[nokeepai/Inkling-2-Mini-Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/tree/v2.1).

**Other fine-tunes of Inkling 2 Mini** may work, but we have not measured any. If you try one, run your own checks.

<details>
<summary>How the evaluated models were merged</summary>

The evaluated models were merged by
[`scripts/v3stack/cand.sh`](https://github.com/nokeepai/inkling-2-mini-unbound/blob/main/scripts/v3stack/cand.sh) in the
GitHub repo, which runs the command above inside a pinned SGLang
container image for torch and safetensors, with `--device cuda` and seed 0.

</details>

## A second, independent check

The numbers above come from the merged weights, served on our own GPU server. We also ran the public benchmarks a
second way: on **Tinker**, Thinking Machines' own hosted service for fine-tuning and running their models. There
the adapter is applied on the fly, without merging and so without our rounding step, on software we did not write.

This matters because it rules out the merge or our serving setup as the reason for the results. Same questions,
same seeds, effort 0.9, the original model against the original plus this adapter. The results
agree: no benchmark came out clearly worse (a few small ones were **inconclusive**, meaning the test was too small
to rule out a larger loss), and average thinking per question fell from 1,010
to 462 tokens.

<details>
<summary>Every benchmark on the hosted service, one row each</summary>

"Difference" is the adapter minus the original, on the same questions. **Non-inferior** means the whole 95%
interval of the difference stays above −2 points: no meaningful loss. **Inconclusive**, as above, means the
test was too small to rule out a larger loss. Scores carry Wilson 95% intervals; differences carry a paired bootstrap
95% interval. "Strict format" maths rejects a correct number written with units, such as "230 miles", which the
original model does more often; read the "number correct" row.

| Benchmark | What it tests | Questions | Original | Original + adapter | Difference, points | Verdict |
|---|---|---|---|---|---|---|
| HumanEval | writing short Python functions | 164 | 98.2% [94.8, 99.4] | 98.8% [95.7, 99.7] | +0.6 pts [-1.2, +2.4] | non-inferior |
| MBPP | writing short Python functions | 257 | 95.3% [92.0, 97.3] | 96.9% [94.0, 98.4] | +1.6 pts [+0.4, +3.1] | non-inferior (interval above zero; no gain claimed) |
| HumanEvalFix | fixing a bug in a function | 164 | 98.2% [94.8, 99.4] | 97.0% [93.1, 98.7] | -1.2 pts [-3.0, +0.0] | inconclusive (interval reaches past the margin) |
| QuixBugs | fixing a bug in a classic algorithm | 31 | 96.8% [83.8, 99.4] | 93.5% [79.3, 98.2] | -3.2 pts [-12.9, +6.5] | inconclusive (interval reaches past the margin) |
| GSM8K, number correct | maths word problems | 1,319 | 96.5% [95.4, 97.4] | 96.5% [95.4, 97.4] | +0.0 pts [-0.8, +0.8] | non-inferior |
| GSM8K, strict format | the same, exact answer format | 1,319 | 69.7% [67.2, 72.2] | 89.5% [87.7, 91.0] | +19.7 pts [+17.4, +22.0] | non-inferior (interval above zero; no gain claimed) |
| MATH-500 | competition maths | 500 | 92.4% [89.7, 94.4] | 93.2% [90.7, 95.1] | +0.8 pts [-0.2, +2.0] | non-inferior |
| MMLU | general knowledge, multiple choice | 2,000 | 89.1% [87.7, 90.4] | 89.9% [88.5, 91.2] | +0.8 pts [-0.2, +1.8] | non-inferior |

</details>

<details>
<summary>How much scores move by chance (repeated runs with different seeds)</summary>

Some benchmarks were run several times per model, each time with a different seed, to show how much a score moves
by chance alone.

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

**Seeds pooled** (every item run once per seed; paired difference with a bootstrap that resamples items as clusters, 10,000 resamples):

| Benchmark | Items × seeds | Original | Prime | Difference, points | Read-out |
|---|---|---|---|---|---|
| HumanEvalFix | 164 × 3 | 470/492 (95.5%) | 472/492 (95.9%) | +0.4 pts [-0.6, +1.6] | non-inferior |
| MBPP | 257 × 3 | 748/771 (97.0%) | 739/771 (95.8%) | -1.2 pts [-2.5, +0.1] | inconclusive (interval reaches past the margin) |


</details>

## Files

| File | What it is |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | The adapter: LoRA rank 32 (its size setting), changing the attention and expert weights. |
| `merge_adapter.py` | The script that merges the adapter into an Inkling 2 Mini checkpoint, with stochastic rounding. |

<details>
<summary>Checksums</summary>

- Adapter file sha256: `a13961f470a8b804b28f1c065c6b770ba6a0c8f449d554292d3f6ea9f8b6fdb9`
- Merged Inkling-2-Mini-Prime, `model-00001` sha256: `823f44bbcbdcb50e790cde840b310c71acca4c2317c9ef806e47fc29f81c827b`
- Merged Unbound v3, `model-00001` sha256: `88e78b0126722a99dcecf26dd7b6cf242860d415f4f74b651e57b18d6abf3b88`

Hugging Face also shows each file's size and sha256 in this repo's file list.

</details>

## Limits

- **Measured on two models only:** the original Inkling 2 Mini and Unbound v2.1.
- **No accuracy gain,** and one possible coding dip, as described under
  [What the adapter does](#what-the-adapter-does).
- **Small debugging test.** The headline debugging result is 15 tasks. The direction
  is clear; the exact percentages are not precise.
- **Not measured:** long documents, images and tool calling. Only English was tested.

## Licence

Apache 2.0, like the base model, Inkling 2 Mini by Thinking Machines Lab, Inc. The base model's
[Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) applies.
