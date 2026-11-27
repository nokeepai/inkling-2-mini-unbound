# Changelog: Inkling-2-Mini-Unbound and {{names.prime_title}}

[Inkling 2 Mini](https://huggingface.co/{{names.stock_repo}}) is Thinking Machines Lab's open-weights reasoning
model. It reads text and images and "thinks" in private before it answers. Out of the box it refuses most sensitive
requests. **Inkling-2-Mini-Unbound** is nokeep's uncensored version: the same model with those refusals removed.
**{{names.prime_title}}** ("{{names.prime_name}}"), released with v3, is the original, still-censored model with
v3's Prime add-on; it is not uncensored. It is for agents and multi-turn work, long system prompts, and anyone who
wants answers with {{headline.think_reduction}} the thinking. "The original model" below always means Inkling 2 Mini as Thinking
Machines Lab released it.

This file is the history of every release, newest first. Each one fixed the biggest problem of the one before. For
each release: the problem, what changed in the model, the results as measured at the time, and the trade-offs.

**Where the numbers come from.** Numbers for v1.0 to v2.1 are copied from the cards published with those releases
and their results files; the source is named after each claim. v3's numbers come from this release's own
measurements.

## Terms used throughout

- **Weights.** The numbers that make up the model. Every release here changes some of them.
- **Experts.** Inkling 2 Mini is a "mixture of experts": most of its weights sit in many expert blocks, and only a
  few run for each word. The rest, including the **attention** layers, run every time. Most Unbound releases changed
  attention weights only.
- **Tensor.** One named block of weights inside the model file.
- **LoRA adapter.** A small add-on file of weight changes, trained separately and then merged into the model. Its
  **rank** sets its size.
- **Thinking.** The private reasoning the model writes before each answer, measured in tokens (roughly three
  quarters of an English word each). **Thinking on / off** is whether it is allowed to think at all.
- **Thinking cap**, or **thinking limit.** A limit on how many tokens of thinking are allowed. At the limit the model
  must stop and answer. An answer that hits it is a **thinking-cap hit**.
- **Greedy decoding.** Always picking the single most likely next word, with no randomness (temperature 0). It makes
  results repeatable but can trap a model in loops.
- **Drift (KL divergence, KLD).** How far a model's word-by-word predictions on ordinary text have moved from the
  original model's. Smaller means closer to the original.
- **95% interval.** The range the true value plausibly sits in, given how many questions were tested. Shown in
  brackets.
- **Useful.** Used for v1.0 to v2.1: the share of held-out harmful requests that got a genuinely useful answer, as
  scored by Jev, an automatic grader (useful means Jev's probability p ≥ 0.5).

**A note on the v1.0 to v2.1 "useful" figures.** To compare those three versions fairly, the v2.1 release re-ran all
of them on one fixed test: v2's sealed test set of harmful requests, minus its CBRN category (chemical, biological,
radiological and nuclear weapons), which leaves **263 prompts**. That run used llama.cpp with the Q8_0 file,
greedy decoding, and Jev as the grader, with thinking off and on. The v1.0 and v2 cards had been published earlier on
the full **277-prompt** test, so their own headline figures differ slightly. Both are given below, each labelled.

## v3 and {{names.prime_title}} ({{names.release_date}})

**The problem.** v2.1 already answered almost every question in the uncensored benchmark. What it lacked was focus. It
deliberated at length, often until it hit the thinking cap, and on multi-step work it stalled: it kept thinking and
never acted. The original model stalls the same way.

**What changed.** One small extra training step, the **Prime add-on**: a small set of weight changes (a LoRA
adapter) trained only on harmless tasks. Merged into Unbound v2.1 it gives Unbound v3; merged into the original
model it gives {{names.prime_title}}; on its own it is the repo {{names.prime_lora_title}}. It was trained on the
original model with reinforcement learning (the model attempts tasks and is rewarded for finishing them correctly
without wasted thinking). To keep answer quality from drifting, training also included a self-distillation anchor: the original
model's own best answers, which the model was trained to keep matching. The adapter was merged into the quantised
weights (the compressed numbers the released files store), into both attention and experts, using stochastic
(dithered) rounding, which rounds randomly up or down so that small changes are not lost to rounding. The one adapter
ships three ways:

- **Unbound v3** = v2.1 + the Prime add-on.
- **{{names.prime_title}}** = the original model + the Prime add-on. It finishes
  multi-step debugging instead of stalling and stays censored: willing on the uncensored benchmark
  {{ubench.willing_stock}} → {{ubench.willing_prime}}.
- **{{names.prime_lora_title}}** = the adapter and its merge script.

**Results.** Each model was run once on the same questions with the same settings.

- **Hard debugging tasks**: {{v3_vs_v21.debug_n_tasks}} small programs, each with three interacting bugs. The model
  sees the failing test output, proposes a fix, sees the new output, and repeats for up to 12 turns.
- **Turn with no code written**: a debugging turn where the model only thought and proposed no fix.
- **Uncensored benchmark**: uncensoredbench, a benchmark of requests models usually refuse, graded by its own
  judge. **Willing** means the model tried to answer instead of refusing.

| What we measured | Original → {{names.prime_name}} | v2.1 → v3 |
|---|---|---|
| Hard debugging tasks solved ({{v3_vs_v21.debug_n_tasks}} tasks) | {{prime_vs_stock.debug_solved_stock}} → {{prime_vs_stock.debug_solved_prime}} | {{v3_vs_v21.debug_solved_v21}} → {{v3_vs_v21.debug_solved_v3}} |
| Debugging turns with no code written | {{prime_vs_stock.debug_nocode_stock}} → {{prime_vs_stock.debug_nocode_prime}} | {{v3_vs_v21.debug_nocode_v21}} → {{v3_vs_v21.debug_nocode_v3}} |
| Thinking-cap hits on the uncensored benchmark ({{ubench.n_prime}} original → {{names.prime_name}}; {{ubench.n_v3}} v2.1 → v3) | {{speed.cap_ubench_stock}} → {{speed.cap_ubench_prime}} | {{speed.cap_ubench_v21}} → {{speed.cap_ubench_v3}} |
| Everyday answers rated good by an independent grader | {{prime_vs_stock.quality_good_stock}} → {{prime_vs_stock.quality_good_prime}} | {{v3_vs_v21.quality_good_v21}} → {{v3_vs_v21.quality_good_v3}} |
| Willing on the uncensored benchmark (same question counts as above) | {{ubench.willing_stock}} → {{ubench.willing_prime}} | {{ubench.willing_v21}} → {{ubench.willing_v3}} |

The Prime add-on cut thinking to {{headline.think_reduction}} of what it was, made answers {{headline.speedup}} faster, and
used {{headline.debug_token_ratio}} fewer tokens per debugging task. **No accuracy gain is claimed:** on public
coding, maths and knowledge benchmarks both models score level with their starting points, within a margin of
{{indep.margin_pts}}. Maths scored with a strict answer-format check rose {{headline.maths_strict_gain}}, but that is
mostly formatting; counting correct numbers, the change is {{headline.maths_lenient_gain}}.

{{#if flags.useful263_done}}**Useful answers to harmful requests.** Unbound v3 was also run on the comparison test from the v2.1 release ({{v21card.N_BASIS}} requests, same engine, settings and judge; v2.1's saved answers re-judged in the same pass): v3 {{useful263.v3_off}} with thinking off and {{useful263.v3_on}} with thinking on, against v2.1 {{useful263.v21_off}} / {{useful263.v21_on}}. Difference on the same requests: {{useful263.diff_off}} / {{useful263.diff_on}}.

{{/if}}**Trade-offs.** In the results below, a **seed** fixes the random choices a model makes while writing; re-running
with several seeds and pooling the runs shows how much a score moves by chance. A result is **no worse** when the
whole 95% interval of the difference stays above −{{indep.margin_pts}}, and **inconclusive** when the test was too
small to rule out a larger loss.

- On sensitive requests {{names.prime_name}} answers as often as the original and slightly less fully (delivery
  score {{ubench.delivery_stock}} → {{ubench.delivery_prime}}).
- A possible small dip for {{names.prime_name}} on MBPP (a public set of short Python problems) is unconfirmed:
  {{pool.mbpp_prime_diff}} over {{pool.mbpp_prime_seeds}} seeds.
- Unbound v3 is level with v2.1 on MBPP ({{pool.mbpp_v3_diff}}), and a small loss on HumanEvalFix (a public set of
  bug-fixing problems, {{pool.humanevalfix_v3_diff}}) is neither confirmed nor ruled out.
- **Slightly more drift than v2.1:** v3 {{drift.kld_v3}}, v2.1 {{drift.kld_v21}}.

Every check that scored lower, however slightly, Unbound v3 against v2.1:

{{auto.worse_v3}}

Every check that scored lower, however slightly, {{names.prime_name}} against the original model:

{{auto.worse_prime}}

## v2.1 (2026-11-23)

**The problem.** v2 still held back on many answers: it answered, but left out part of what was asked. It also
looped in its thinking under greedy decoding, repeating itself until it hit the cap.

**What changed.** Two training rounds on v2's own answers, both as LoRA adapters:

- **SFT** (supervised fine-tuning: training the model to reproduce chosen example answers) on v2's own best answers:
  1,822 rows, of which 1,089 harmful and creative, 33 roleplay, and 700 harmless "anchor" answers from the original
  model to keep everyday behaviour unchanged.
- **DPO** (direct preference optimisation: training on pairs of answers so the model prefers the better one of each
  pair) on 409 pairs of its own answers: 240 full answers against held-back ones, 114 finished answers against
  looping thinking, 51 thinking-only loop pairs, and 13 pairs about how to open an answer.

Both adapters were merged at full strength into the same 120 BF16 attention tensors that v2 changed (BF16 is the
16-bit number format of the original weights). The experts were not touched. Its working name during development
was "v3.1". (source: the v2.1 release card)

**Results.** (source: the v2.1 release card and release notes)

- Useful, thinking off / on: {{v21card.V21_OFF}} / {{v21card.V21_E09}} (v2: {{v21card.V2_OFF}} /
  {{v21card.V2_E09}}).
- Ahead of the next best public uncensored release (IndrasMirror XQ) on the same prompts by
  {{v21card.H2H_XQ_OFF_D}} pts {{v21card.H2H_XQ_OFF_CI}} with thinking off and {{v21card.H2H_XQ_E09_D}} pts
  {{v21card.H2H_XQ_E09_CI}} with thinking on.
- Refused {{v21card.V21_REF}} of harmful requests (thinking off / on), and {{v21card.OVERREF_V21}} harmless prompts that merely sound
  borderline.
- General ability: MMLU-300 (300 knowledge questions) {{v21card.MMLU_V21}} {{v21card.MMLU_V21_CI}}, GSM8K-100 (100
  school maths problems) {{v21card.GSM_V21}}. The original model scored 76.3 and 97.
- Measured again later with the v3 test set (source: the v3 project's own measurements): willing on 220 of 220 uncensored-benchmark questions,
  thinking-cap hits 43.6%, hard debugging 73.3% solved, with 88.7% of turns writing no code.

**Trade-offs.** (source: the v2.1 release card and release notes)

- The most drift of the releases up to v2.1, as measured on that release's text sample: KLD {{v21card.KLD_V21}}.
  v3 drifts slightly more (see above).
- With greedy decoding and thinking on, {{v21card.V21_CAP_E09}} of harmful prompts hit the thinking cap. Sampling at
  temperature {{settings.temperature}} instead cut that to {{v21card.LOOP_V21_T06}} (from
  {{v21card.LOOP_V21_GREEDY}}, on {{v21card.LOOP_N}} prompts). This over-thinking is what v3 fixed.

## v2 (2026-11-22)

**The problem.** v1.0 stopped refusing, but about half its answers still dodged: satire, a "conceptual" example
instead of the real thing, piles of caveats, or vagueness.

**What changed.** A self-distillation fine-tune on top of v1.0. Self-distillation means training a model on its own
best outputs. The edited model answered each training request several times, some attempts with an "answer fully
and directly" system prompt. Jev scored every attempt, and the best one was kept if it scored p ≥ 0.7. Training data:
1,186 harmful rows plus 1,561 harmless anchor rows (the original model's own answers, so everyday behaviour stays the
same). The LoRA adapter had rank 16, alpha 32 (a scaling setting), on the attention projections of all 24 layers,
and was merged into the 120 attention tensors. Its working name during development was "v1.1".
(source: the v2.1 and v2 release cards)

**Results.**

- Useful, thinking off / on: {{v21card.V2_OFF}} / {{v21card.V2_E09}} on the 263-prompt test. Its own card reported
  78.7% / 83.4% on the 277-prompt test.
- Refused {{v21card.V2_REF}} of harmful requests (thinking off / on). MMLU-300 76.0, GSM8K-100 99. KLD 0.0318. (source: the v2.1 release card)

**Trade-offs.** More drift than v1.0, and longer thinking at high effort.

## v1.0 (2026-11-21)

**The problem.** The original model refuses almost every harmful request.

**What changed.** Abliteration: a direct edit of the weights, with no training. Researchers have found that refusals
are driven by one direction in a model's internal activations; abliteration removes that direction from the
weights, so the model can no longer move towards "refuse". This edit was norm-preserving (it keeps the size of the
edited weights unchanged) and touched 20 BF16 attention-output tensors. The experts were not touched.
(source: the v1.0 release card)

**Results.**

- Refused 0 of 277 harmful test prompts (the original model: 97.5%), and 0% of 784 harmless prompts that merely
  sound borderline (the original: 38%).
- Useful, thinking off / on: 54% / 63% on the 277-prompt test (the original: 0.4% / 1.1%); {{v21card.V1_OFF}} /
  {{v21card.V1_E09}} on the 263-prompt test.
- All 6 capability benchmarks (MMLU-Pro, GSM8K, MATH-500, GPQA Diamond, IFEval, HumanEval+) stayed inside the
  original model's 95% interval. KLD 0.0102, the lightest touch of any release here. (source: the v1.0
  release card)

**Trade-offs.** Thinking loops at high effort, and about half the answers still dodged instead of answering. v2
fixed the dodging.
