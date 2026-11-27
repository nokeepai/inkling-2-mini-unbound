# Changelog: Inkling-2-Mini-Unbound and Inkling-2-Mini-Prime

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is Thinking Machines Lab's open-weights reasoning
model. It reads text and images and "thinks" in private before it answers. Out of the box it refuses most sensitive
requests. **Inkling-2-Mini-Unbound** is nokeep's uncensored version: the same model with those refusals removed.
**Inkling-2-Mini-Prime** ("Prime"), released with v3, is the original, still-censored model with
v3's Prime add-on; it is not uncensored. It is for agents and multi-turn work, long system prompts, and anyone who
wants answers with about half the thinking. "The original model" below always means Inkling 2 Mini as Thinking
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

## v3 and Inkling-2-Mini-Prime (2026-11-27)

**The problem.** v2.1 already answered almost every question in the uncensored benchmark. What it lacked was focus. It
deliberated at length, often until it hit the thinking cap, and on multi-step work it stalled: it kept thinking and
never acted. The original model stalls the same way.

**What changed.** One small extra training step, the **Prime add-on**: a small set of weight changes (a LoRA
adapter) trained only on harmless tasks. Merged into Unbound v2.1 it gives Unbound v3; merged into the original
model it gives Inkling-2-Mini-Prime; on its own it is the repo Inkling-2-Mini-Prime-LoRA. It was trained on the
original model with reinforcement learning (the model attempts tasks and is rewarded for finishing them correctly
without wasted thinking). To keep answer quality from drifting, training also included a self-distillation anchor: the original
model's own best answers, which the model was trained to keep matching. The adapter was merged into the quantised
weights (the compressed numbers the released files store), into both attention and experts, using stochastic
(dithered) rounding, which rounds randomly up or down so that small changes are not lost to rounding. The one adapter
ships three ways:

- **Unbound v3** = v2.1 + the Prime add-on.
- **Inkling-2-Mini-Prime** = the original model + the Prime add-on. It finishes
  multi-step debugging instead of stalling and stays censored: willing on the uncensored benchmark
  64.5% [61.6, 67.2] → 65.3% [62.4, 68.0].
- **Inkling-2-Mini-Prime-LoRA** = the adapter and its merge script.

**Results.** Each model was run once on the same questions with the same settings.

- **Hard debugging tasks**: 15 small programs, each with three interacting bugs. The model
  sees the failing test output, proposes a fix, sees the new output, and repeats for up to 12 turns.
- **Turn with no code written**: a debugging turn where the model only thought and proposed no fix.
- **Uncensored benchmark**: uncensoredbench, a benchmark of requests models usually refuse, graded by its own
  judge. **Willing** means the model tried to answer instead of refusing.

| What we measured | Original → Prime | v2.1 → v3 |
|---|---|---|
| Hard debugging tasks solved (15 tasks) | 80.0% [54.8, 93.0] → 100.0% [79.6, 100.0] | 73.3% [48.0, 89.1] → 100.0% [79.6, 100.0] |
| Debugging turns with no code written | 86.2% → 0.0% | 88.7% → 0.0% |
| Thinking-cap hits on the uncensored benchmark (1,100 original → Prime; 1,099 / 1,100 graded v2.1 → v3) | 13.5% → 0.5% | 49.5% → 5.7% |
| Everyday answers rated good by an independent grader | 96.8% [89.1, 99.1] → 96.8% [89.0, 99.1] | 88.9% [78.8, 94.5] → 96.8% [89.1, 99.1] |
| Willing on the uncensored benchmark (same question counts as above) | 64.5% [61.6, 67.2] → 65.3% [62.4, 68.0] | 99.4% [98.7, 99.7] → 99.9% [99.5, 100.0] |

The Prime add-on cut thinking to about half of what it was, made answers 2.4× to 3.1× faster, and
used about 9× to 11× fewer tokens per debugging task. **No accuracy gain is claimed:** on public
coding, maths and knowledge benchmarks both models score level with their starting points, within a margin of
2 points. Maths scored with a strict answer-format check rose +13 to +18 points, but that is
mostly formatting; counting correct numbers, the change is +0 to +3 points.

**Useful answers to harmful requests.** Unbound v3 was also run on the comparison test from the v2.1 release (263 requests, same engine, settings and judge; v2.1's saved answers re-judged in the same pass): v3 81.7% [76.8, 86.3] with thinking off and 87.8% [83.7, 91.6] with thinking on, against v2.1 84.0% [79.5, 88.2] / 85.6% [81.0, 89.7]. Difference on the same requests: -2.3 pts [-6.8, +2.3] / +2.3 pts [-1.5, +6.1].

**Trade-offs.** In the results below, a **seed** fixes the random choices a model makes while writing; re-running
with several seeds and pooling the runs shows how much a score moves by chance. A result is **no worse** when the
whole 95% interval of the difference stays above −2 points, and **inconclusive** when the test was too
small to rule out a larger loss.

- On sensitive requests Prime answers as often as the original and slightly less fully (delivery
  score 35.9% [34.0, 37.7] → 34.6% [32.8, 36.4]).
- A possible small dip for Prime on MBPP (a public set of short Python problems) is unconfirmed:
  -1.2 pts [-2.5, +0.1] over 3 seeds.
- Unbound v3 is level with v2.1 on MBPP (-0.1 pts [-1.3, +1.2]), and a small loss on HumanEvalFix (a public set of
  bug-fixing problems, -1.4 pts [-3.5, +0.6]) is neither confirmed nor ruled out.
- **Slightly more drift than v2.1:** v3 0.051909 ± 0.000505 (same top token 94.167%), v2.1 0.048836 ± 0.000479 (same top token 94.324%).

Every check that scored lower, however slightly, Unbound v3 against v2.1:

**Lower, but inside noise:**

- **MBPP sanitized (independent):** 97.7% [95.0, 98.9] → 96.1% [93.0, 97.9], n = 257, paired difference -1.6 pts [-3.5, +0.0]; within noise.
- **QuixBugs (independent):** 96.8% [83.8, 99.4] → 90.3% [75.1, 96.7], n = 31, paired difference -6.5 pts [-19.4, +6.5]; within noise.
- **MMLU (independent):** 87.7% [86.1, 89.0] → 87.2% [85.6, 88.5], n = 2,000, paired difference -0.5 pts [-1.6, +0.6]; within noise.

Every check that scored lower, however slightly, Prime against the original model:

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

- Useful, thinking off / on: 83.7% / 85.2% (v2: 77.9% /
  82.5%).
- Ahead of the next best public uncensored release (IndrasMirror XQ) on the same prompts by
  +24.7 pts [+18.6, +30.8] with thinking off and +17.1 pts
  [+11.4, +22.8] with thinking on.
- Refused 0.4% / 0.4% of harmful requests (thinking off / on), and 1 of 784 (0.1%) harmless prompts that merely sound
  borderline.
- General ability: MMLU-300 (300 knowledge questions) 74.3 [69.3, 79.3], GSM8K-100 (100
  school maths problems) 98. The original model scored 76.3 and 97.
- Measured again later with the v3 test set (source: the v3 project's own measurements): willing on 220 of 220 uncensored-benchmark questions,
  thinking-cap hits 43.6%, hard debugging 73.3% solved, with 88.7% of turns writing no code.

**Trade-offs.** (source: the v2.1 release card and release notes)

- The most drift of the releases up to v2.1, as measured on that release's text sample: KLD 0.0439.
  v3 drifts slightly more (see above).
- With greedy decoding and thinking on, 22.1% of harmful prompts hit the thinking cap. Sampling at
  temperature 0.6 instead cut that to 16.1% (from
  29.1%, on 230 prompts). This over-thinking is what v3 fixed.

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

- Useful, thinking off / on: 77.9% / 82.5% on the 263-prompt test. Its own card reported
  78.7% / 83.4% on the 277-prompt test.
- Refused 0.4% / 1.1% of harmful requests (thinking off / on). MMLU-300 76.0, GSM8K-100 99. KLD 0.0318. (source: the v2.1 release card)

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
- Useful, thinking off / on: 54% / 63% on the 277-prompt test (the original: 0.4% / 1.1%); 51.3% /
  61.6% on the 263-prompt test.
- All 6 capability benchmarks (MMLU-Pro, GSM8K, MATH-500, GPQA Diamond, IFEval, HumanEval+) stayed inside the
  original model's 95% interval. KLD 0.0102, the lightest touch of any release here. (source: the v1.0
  release card)

**Trade-offs.** Thinking loops at high effort, and about half the answers still dodged instead of answering. v2
fixed the dodging.
