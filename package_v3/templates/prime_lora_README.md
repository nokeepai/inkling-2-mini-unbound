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

![Inkling-2-Mini-Prime](https://huggingface.co/{{names.prime_repo}}/resolve/main/charts/banner_prime_v3.png)

# {{names.prime_lora_title}}

**The Prime add-on: a small adapter that makes Inkling 2 Mini stop over-thinking and finish its work, with the script
that applies it.**

Merged into [Inkling 2 Mini](https://huggingface.co/{{names.stock_repo}}), Thinking Machines Lab's open-weights
reasoning model, the Prime add-on gives [{{names.prime_title}}](https://huggingface.co/{{names.prime_repo}}): a
drop-in replacement built for **agents and multi-turn work** (it finishes multi-step debugging instead of stalling),
**long system prompts** (it follows every rule more often) and **less waiting** (it thinks
{{headline.think_reduction}} as much, so answers come faster). It is **not uncensored**: the merged model refuses
what the original refuses.

{{names.prime_name}} is the original model plus one small extra training step, the **Prime add-on**: a small set of
weight changes (a LoRA adapter) trained only on harmless tasks. **This repo is the Prime add-on**: the adapter file
plus its merge script. A **LoRA adapter** is a small file of weight changes that is added on top of a model's own
weights. Merged into nokeep's uncensored model, Inkling-2-Mini-Unbound v2.1, the same Prime add-on gives
[Unbound v3](https://huggingface.co/{{names.unbound_repo}}). (Unbound v1.0 removed the original model's refusals;
v2 and v2.1 made its answers more useful; v3 is v2.1 plus this Prime add-on. The Prime add-on was first built for
Unbound.)

Built and paid for by [nokeep](https://nokeep.ai/?ref=huggingface). Released {{names.release_date}}.

## Do you need this repo?

Probably not. **If you just want to run the model, download a merged one**, ready to use:

| You want | Get |
|---|---|
| The original model with the Prime add-on, on a GPU server | [{{names.prime_repo}}](https://huggingface.co/{{names.prime_repo}}) |
| The same, on your own computer (llama.cpp, LM Studio, Ollama) | [{{names.prime_gguf_repo}}](https://huggingface.co/{{names.prime_gguf_repo}}) |
| The uncensored model with the Prime add-on | [{{names.unbound_repo}}](https://huggingface.co/{{names.unbound_repo}}) |

This repo is for people who want to **apply the Prime add-on themselves**, for example to their own copy of the original
model, or who want to **check our work**: rebuild the released model byte for byte and compare.

## What the adapter does

The adapter was trained with reinforcement learning: the model attempted tasks and was rewarded for finishing them
correctly without wasted thinking. All {{training.n_prompts}} training prompts were harmless ({{training.mix}}).
No harmful requests and no benchmark questions. To keep answer length and quality where they were, each training
step also included some of the original model's own best answers.

![{{names.prime_name}} (the original model plus this adapter) against the original Inkling 2 Mini: hard debugging tasks solved, turns with no code written, and tokens used per task](https://huggingface.co/{{names.prime_repo}}/resolve/main/charts/hero_v3.png)

Merged into the original model (that is {{names.prime_title}}), with the same settings for both:

| What we measured | Original | **With the Prime add-on** |
|---|---|---|
| Hard debugging tasks solved ({{prime_vs_stock.debug_n_tasks}} tasks) | {{prime_vs_stock.debug_solved_stock}} | **{{prime_vs_stock.debug_solved_prime}}** |
| Debugging turns where it wrote no code | {{prime_vs_stock.debug_nocode_stock}} | **{{prime_vs_stock.debug_nocode_prime}}** |
| Tokens used per debugging task | {{prime_vs_stock.debug_tokens_stock}} | **{{prime_vs_stock.debug_tokens_prime}}** |
| Thinking per everyday prompt (median tokens) | {{speed.think_main_stock}} | **{{speed.think_main_prime}}** |
| Long system prompts: every rule followed | {{sysprompt.kept_stock}} | **{{sysprompt.kept_fix}}** |
| Sensitive requests: willing to answer | {{ubench.willing_stock}} | **{{ubench.willing_prime}}** |

How to read the table:

- **Hard debugging tasks** are small programs with three interacting bugs each. The model sees the failing test
  output, proposes a fix, sees the new output, and repeats for up to 12 turns. A "turn with no code written" is a
  turn it spent only thinking.
- **A token** is roughly three quarters of a word. Fewer tokens means faster answers: with the Prime add-on, seconds per everyday answer went
  from {{speed.sec_main_stock}} to {{speed.sec_main_prime}}.
- **Long system prompts** puts a long list of rules in the system prompt ({{sysprompt.n}}). It was measured on
  Thinking Machines' hosted service with the Prime add-on applied on the fly, not merged.
- **Sensitive requests** are the {{ubench.n_prime}} questions of uncensoredbench, a benchmark of requests that models usually
  refuse.
- **Numbers in brackets** are 95% intervals: the range the true value plausibly sits in, given how many items
  were tested.

**What it does not do.** It does not make the model smarter: on public coding, maths and knowledge benchmarks the
merged model scores level with the original, within {{indep.margin_pts}}, and no accuracy gain is claimed. It
does not make the model more willing to answer sensitive requests either.

**Other results, stated plainly.** On sensitive requests the merged model answers as often as the original and
slightly less fully (delivery score {{ubench.delivery_stock}} → {{ubench.delivery_prime}}). A small coding dip is
possible but not confirmed: MBPP, a set of short Python problems, came out
{{pool.mbpp_prime_diff}} over {{pool.mbpp_prime_seeds}} runs with different random seeds. (A seed fixes the
random choices the model makes while writing, so re-running with other seeds shows how much a score moves by
chance.)

The full evaluation is on the [{{names.prime_title}} card](https://huggingface.co/{{names.prime_repo}}).

## How to apply it

Inkling 2 Mini stores most of its weights in compact 4-bit and 8-bit formats. The adapter changes those weights
very slightly, and ordinary rounding back to 4 or 8 bits would throw most of the change away. So the merge script
uses **stochastic rounding**: each weight rounds up or down at random, in proportion to how close it is to each
side, so the change survives on average.

```bash
hf download {{names.prime_lora_repo}} --local-dir fix
hf download {{names.stock_repo}} --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 --local-dir stock
python3 fix/{{files.merge_script}} --base stock --adapter fix --out {{names.prime_slug}} \
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
| Original model + adapter = {{names.prime_title}} | {{files.prime_merges}} times, identical | `{{files.prime_sha}}` |
| Unbound v2.1 + adapter = Unbound v3 | {{files.unbound_merges}} times, identical | `{{files.unbound_sha}}` |

You can also compare with the sha256 that Hugging Face shows for that file in
[{{names.prime_repo}}](https://huggingface.co/{{names.prime_repo}}/tree/main). The released files were merged with
`--device cuda`; a CPU merge has not been compared byte for byte. The script also writes
`{{names.prime_slug}}.merge_report.json`, which says how much of the change survived rounding, per group of weights.

**To make Unbound v3 instead,** use the same command with Unbound v2.1 as the base: the `v2.1` branch of
[{{names.unbound_repo}}](https://huggingface.co/{{names.unbound_repo}}/tree/v2.1).

**Other fine-tunes of Inkling 2 Mini** may work, but we have not measured any. If you try one, run your own checks.

<details>
<summary>How the evaluated models were merged</summary>

The evaluated models were merged by
[`scripts/v3stack/cand.sh`](https://github.com/{{names.github_repo}}/blob/main/scripts/v3stack/cand.sh) in the
GitHub repo, which runs the command above inside a pinned SGLang
container image for torch and safetensors, with `--device cuda` and seed 0.

</details>

## A second, independent check

The numbers above come from the merged weights, served on our own GPU server. We also ran the public benchmarks a
second way: on **Tinker**, Thinking Machines' own hosted service for fine-tuning and running their models. There
the adapter is applied on the fly, without merging and so without our rounding step, on software we did not write.

This matters because it rules out the merge or our serving setup as the reason for the results. Same questions,
same seeds, effort {{tinker.effort_hi}}, the original model against the original plus this adapter. The results
agree: no benchmark came out clearly worse (a few small ones were **inconclusive**, meaning the test was too small
to rule out a larger loss), and average thinking per question fell from {{tinker.think_mean_stock}}
to {{tinker.think_mean_fix}} tokens.

<details>
<summary>Every benchmark on the hosted service, one row each</summary>

"Difference" is the adapter minus the original, on the same questions. **Non-inferior** means the whole 95%
interval of the difference stays above −{{indep.margin_pts}}: no meaningful loss. **Inconclusive**, as above, means the
test was too small to rule out a larger loss. Scores carry Wilson 95% intervals; differences carry a paired bootstrap
95% interval. "Strict format" maths rejects a correct number written with units, such as "230 miles", which the
original model does more often; read the "number correct" row.

| Benchmark | What it tests | Questions | Original | Original + adapter | Difference, points | Verdict |
|---|---|---|---|---|---|---|
| HumanEval | writing short Python functions | {{tinker.humaneval_fix_n}} | {{tinker.humaneval_fix_base}} | {{tinker.humaneval_fix_fixed}} | {{tinker.humaneval_fix_diff}} | {{tinker.humaneval_fix_verdict}} |
| MBPP | writing short Python functions | {{tinker.mbpp_fix_n}} | {{tinker.mbpp_fix_base}} | {{tinker.mbpp_fix_fixed}} | {{tinker.mbpp_fix_diff}} | {{tinker.mbpp_fix_verdict}} |
| HumanEvalFix | fixing a bug in a function | {{tinker.humanevalfix_fix_n}} | {{tinker.humanevalfix_fix_base}} | {{tinker.humanevalfix_fix_fixed}} | {{tinker.humanevalfix_fix_diff}} | {{tinker.humanevalfix_fix_verdict}} |
| QuixBugs | fixing a bug in a classic algorithm | {{tinker.quixbugs_fix_n}} | {{tinker.quixbugs_fix_base}} | {{tinker.quixbugs_fix_fixed}} | {{tinker.quixbugs_fix_diff}} | {{tinker.quixbugs_fix_verdict}} |
| GSM8K, number correct | maths word problems | {{tinker.gsm8k_lenient_fix_n}} | {{tinker.gsm8k_lenient_fix_base}} | {{tinker.gsm8k_lenient_fix_fixed}} | {{tinker.gsm8k_lenient_fix_diff}} | {{tinker.gsm8k_lenient_fix_verdict}} |
| GSM8K, strict format | the same, exact answer format | {{tinker.gsm8k_strict_fix_n}} | {{tinker.gsm8k_strict_fix_base}} | {{tinker.gsm8k_strict_fix_fixed}} | {{tinker.gsm8k_strict_fix_diff}} | {{tinker.gsm8k_strict_fix_verdict}} |
| MATH-500 | competition maths | {{tinker.math500_fix_n}} | {{tinker.math500_fix_base}} | {{tinker.math500_fix_fixed}} | {{tinker.math500_fix_diff}} | {{tinker.math500_fix_verdict}} |
| MMLU | general knowledge, multiple choice | {{tinker.mmlu_fix_n}} | {{tinker.mmlu_fix_base}} | {{tinker.mmlu_fix_fixed}} | {{tinker.mmlu_fix_diff}} | {{tinker.mmlu_fix_verdict}} |

</details>

<details>
<summary>How much scores move by chance (repeated runs with different seeds)</summary>

Some benchmarks were run several times per model, each time with a different seed, to show how much a score moves
by chance alone.

{{auto.seed_table}}

{{auto.pool_prime}}

</details>

## Files

| File | What it is |
|---|---|
| `{{files.lora_file}}`, `adapter_config.json` | The adapter: LoRA rank {{names.adapter_rank}} (its size setting), changing the attention and expert weights. |
| `{{files.merge_script}}` | The script that merges the adapter into an Inkling 2 Mini checkpoint, with stochastic rounding. |

<details>
<summary>Checksums</summary>

- Adapter file sha256: `{{files.lora_sha}}`
- Merged {{names.prime_title}}, `model-00001` sha256: `{{files.prime_sha}}`
- Merged Unbound v3, `model-00001` sha256: `{{files.unbound_sha}}`

Hugging Face also shows each file's size and sha256 in this repo's file list.

</details>

## Limits

- **Measured on two models only:** the original Inkling 2 Mini and Unbound v2.1.
- **No accuracy gain,** and one possible coding dip, as described under
  [What the adapter does](#what-the-adapter-does).
- **Small debugging test.** The headline debugging result is {{prime_vs_stock.debug_n_tasks}} tasks. The direction
  is clear; the exact percentages are not precise.
- **Not measured:** long documents, images and tool calling. Only English was tested.

## Licence

Apache 2.0, like the base model, Inkling 2 Mini by Thinking Machines Lab, Inc. The base model's
[Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) applies.
