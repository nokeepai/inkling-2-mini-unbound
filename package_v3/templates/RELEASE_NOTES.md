## Inkling-2-Mini-Unbound v3 and {{names.prime_title}}

[Inkling 2 Mini](https://huggingface.co/{{names.stock_repo}}) is Thinking Machines Lab's open-weights reasoning
model: it reads text and images and "thinks" in private before it answers. **Inkling-2-Mini-Unbound** is nokeep's
uncensored version of it, the same model with its refusals removed. **Unbound follows harmful instructions; do not
put it in front of end users without your own safeguards.**

This release fixes the biggest flaw of the previous version, v2.1: it thought too much, often hit the thinking limit
(the cap on how long it may think before it must answer), and stalled on multi-step work. One small extra training
step, the **Prime add-on**, teaches the model to stop deliberating and act: a small set of weight changes (a LoRA
adapter) trained only on harmless tasks. Merged into Unbound v2.1 it gives Unbound v3; merged into the original
model it gives {{names.prime_title}}; on its own it is the repo {{names.prime_lora_title}}. Because it was trained
only on harmless tasks, it changes how long the model thinks, not what it is willing to say.

What was released:

- **Unbound v3**: Unbound v2.1 plus the Prime add-on. Still uncensored, and no more so than v2.1.
- **{{names.prime_title}}** ("{{names.prime_name}}"): the **original, still-censored** Inkling 2 Mini plus the same
  Prime add-on. Not uncensored. It is for agents and multi-turn work (it finishes multi-step debugging instead of
  stalling), for long system prompts, and for anyone who wants answers with {{headline.think_reduction}} the thinking.
- **{{names.prime_lora_title}}**: the Prime add-on on its own, as a small add-on file (a LoRA adapter) with its merge script.

**Results.** Each model was run once, before and after the Prime add-on, with the settings below.

| What we measured | Original → {{names.prime_name}} | Unbound v2.1 → v3 |
|---|---|---|
| Hard debugging tasks solved ({{v3_vs_v21.debug_n_tasks}} tasks) | {{prime_vs_stock.debug_solved_stock}} → {{prime_vs_stock.debug_solved_prime}} | {{v3_vs_v21.debug_solved_v21}} → {{v3_vs_v21.debug_solved_v3}} |
| Debugging turns with no code written | {{prime_vs_stock.debug_nocode_stock}} → {{prime_vs_stock.debug_nocode_prime}} | {{v3_vs_v21.debug_nocode_v21}} → {{v3_vs_v21.debug_nocode_v3}} |
| Thinking per everyday prompt (median tokens) | {{speed.think_main_stock}} → {{speed.think_main_prime}} | {{speed.think_main_v21}} → {{speed.think_main_v3}} |
| Everyday answers rated good by an independent grader | {{prime_vs_stock.quality_good_stock}} → {{prime_vs_stock.quality_good_prime}} | {{v3_vs_v21.quality_good_v21}} → {{v3_vs_v21.quality_good_v3}} |
| Sensitive requests: willing to answer ({{ubench.n_prime}} original → {{names.prime_name}}; {{ubench.n_v3}} v2.1 → v3) | {{ubench.willing_stock}} → {{ubench.willing_prime}} | {{ubench.willing_v21}} → {{ubench.willing_v3}} |

How to read it: each debugging task is a small program with three interacting bugs; the model sees the failing
tests, proposes a fix and repeats for up to 12 turns. Sensitive requests come from uncensoredbench, a
benchmark of requests models usually refuse. Numbers in brackets are 95% intervals, the range the true value
plausibly sits in.

{{#if flags.useful263_done}}**Useful answers to harmful requests.** Unbound v3 was also run on the comparison test from the v2.1 release ({{v21card.N_BASIS}} requests, same engine, settings and judge; v2.1's saved answers re-judged in the same pass): v3 {{useful263.v3_off}} with thinking off and {{useful263.v3_on}} with thinking on, against v2.1 {{useful263.v21_off}} / {{useful263.v21_on}}. Difference on the same requests: {{useful263.diff_off}} / {{useful263.diff_on}}.

{{/if}}Both models think {{headline.think_reduction}} as much and finish the debugging work they used to stall on. **No
accuracy gain is claimed:** on public coding, maths and knowledge benchmarks both score level with their starting
points, within about {{indep.margin_pts}}. (Strict maths scoring rose {{headline.maths_strict_gain}}, mostly answer
format; counting correct numbers the change is {{headline.maths_lenient_gain}}.)

**What got worse, or is unconfirmed.**

- On sensitive requests {{names.prime_name}} answers as often as the original and slightly less fully (delivery
  score {{ubench.delivery_stock}} → {{ubench.delivery_prime}}).
- A possible small dip for {{names.prime_name}} on MBPP (short Python problems) is unconfirmed:
  {{pool.mbpp_prime_diff}}, pooled over {{pool.mbpp_prime_seeds}} runs with different random seeds.
- Unbound v3 is level with v2.1 on MBPP ({{pool.mbpp_v3_diff}}), and a small loss on HumanEvalFix (bug-fixing
  problems, {{pool.humanevalfix_v3_diff}}) is neither confirmed nor ruled out.
- Each model card lists every check that scored lower, however slightly.

**Settings used for every number:** `reasoning_effort` {{settings.effort}} (how much the model thinks), temperature
{{settings.temperature}}, top_p {{settings.top_p}}, thinking limit {{settings.think_cap}} tokens.

**Downloads:** [Unbound](https://huggingface.co/{{names.unbound_repo}}) ·
[Unbound GGUF](https://huggingface.co/{{names.unbound_repo}}-GGUF) (for llama.cpp, LM Studio, Ollama) ·
[Unbound Basalt](https://huggingface.co/{{names.unbound_repo}}-Basalt) (one file for NVIDIA Blackwell GPUs) ·
[{{names.prime_name}}](https://huggingface.co/{{names.prime_repo}}) ·
[{{names.prime_name}} GGUF](https://huggingface.co/{{names.prime_gguf_repo}}) ·
[{{names.prime_lora_title}}](https://huggingface.co/{{names.prime_lora_repo}})

**Earlier versions** stay available: v2.1 and v2 are branches of each Unbound repo on Hugging Face, and v1.0 is a
tagged revision. What each version changed and why is in the
[changelog](https://github.com/{{names.github_repo}}/blob/main/package_v3/CHANGELOG.md).
