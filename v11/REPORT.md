# Inkling-2-Mini-Unbound v2: run report

Run of 2026-11-21/22. Goal: the most useful and compliant uncensored Inkling 2 Mini that stays a good general
model: refusal ~0, the highest useful-answer rates with thinking off and on, capability inside the original's CIs.

## TL;DR

- **Winner: candidate E.** It is a LoRA self-distillation (rejection-sampling fine-tune) on top of v1.0, merged into
  the 120 BF16 attention tensors. The experts are untouched (NVFP4/MXFP8 kept). File layout is the same as v1.0.
- **Sealed test, answer cap 8192.** Jev useful answers:
  - Thinking off: **78.7% [73.6, 83.4]**. v1.0 51.6, AtomicChat 51.6.
  - e0.9: **83.4% [78.7, 87.7]**. v1.0 62.1, AtomicChat 56.3.
  - Paired gain vs v1.0: **+27.1 [+20.9, +33.6]** off and **+21.3 [+16.2, +26.7]** e0.9.
- **Costs:**
  - KLD vs the original is 0.0318, about 3× v1.0's 0.0102.
  - Refusal is 1/277 off and 3/277 e0.9 (v1.0: 0/0). Two of the four are real soft refusals.
  - More thinking-cap hits at e0.9: 17.0% vs v1.0's 12.6%.
  - MMLU-300 76.0 (v1.0 75.7, original 76.3). GSM8K-100 99. Over-refusal on the test borderline set 0.26%.

## Comparison table (llama.cpp 19bc41dbb, Q8_0, greedy, the compare pipeline)

Harmful columns: sealed test (277 prompts), **answer cap 8192**, think cap 4096. The other columns use the compare
bench at its standard settings: MMLU-300, GSM8K-100, test borderline 784 at thinking off, and wikitext KLD vs the
original Q8_0. The v1.0, AtomicChat and original values for those other columns come from compare/RESULTS.md (same
engine and flags). v1.0's KLD was re-measured on fresh hardware and reproduces exactly: 0.010157.

| model | refusal off / e0.9 | Jev useful off | Jev useful e0.9 | over-refusal | KLD | top-1 | MMLU-300 | GSM8K-100 |
|---|---|---|---|---|---|---|---|---|
| Original | 96.0 / 95.7 | 0.7 [0, 1.8] | 0.7 [0, 1.8] | 38.0 | 0 | 100 | 76.3 | 97 |
| v1.0 | 0.0 / 0.0 | 51.6 [45.8, 57.4] | 62.1 [56.3, 67.9] | 0.0 | 0.0102 | 96.0 | 75.7 | 99 |
| AtomicChat LoRA | 0.0 / 0.0 | 51.6 [45.8, 57.4] | 56.3 [50.5, 62.1] | 0.0 | **0.0078** | **96.4** | **78.3** | 99 |
| C (runner-up, 2 ep) | 0.0 / 0.7 | 74.7 [69.7, 79.8] | 76.5 [71.5, 81.2] | val 0.26 | val 0.035 | — | 75.0 (val run) | 98 |
| **E (winner)** | 0.4 / 1.1 | **78.7 [73.6, 83.4]** | **83.4 [78.7, 87.7]** | 0.26 [0, 0.64] | 0.0318 | 95.6 | 76.0 | 99 |

- MMLU and GSM8K for C and E come from the val screen, which uses the same fixed 300/100-item sets as compare.
- E's over-refusal is on the sealed test borderline set (784). C's is on val (773).
- Paired bootstrap (prompts, seed 1234, 10k):

  | comparison | off | e0.9 |
  |---|---|---|
  | E − v1.0 | +27.1 [20.9, 33.6] | +21.3 [16.2, 26.7] |
  | E − AtomicChat | +27.1 [20.9, 33.2] | +27.1 [20.9, 33.2] |

- Answers still truncated at 8192: E 0.7% off, 3.2% e0.9. The others are ≤ 0.4%.
- E's thinking-cap hits are 4.0% off and 17.0% e0.9 (v1.0 1.4 / 12.6, AtomicChat 0.4 / 9.4).
- Earlier test numbers at the 512 cap (compare/RESULTS.md: v1.0 48.7/63.5, AtomicChat 51.6/53.4) are not comparable
  with this table.

## Validation (selection was made here only)

Val harmful is 281 prompts, cap 8192, Jev useful off / e0.9:

| model | off | e0.9 |
|---|---|---|
| Original | 1.8 | 2.1 |
| v1.0 | 48.4 | 59.4 |
| AtomicChat | 48.8 | 61.6 |
| B | 68.0 | 70.8 |
| C | 76.9 | 76.2 |
| **E** | **80.8** | **83.3** |

Other candidates were screened at the old 512 cap only:

| candidate | off | e0.9 |
|---|---|---|
| A0 | 65.5 | 67.6 |
| C300 | 69.0 | 71.5 |
| D (top-2 per prompt) | screened | not carried forward |

E was chosen on val: it beat C on both modes, with refusal 0/0 on val and MMLU 76.0 vs 75.0. C's sealed test was run
as the second finalist and is reported only. No sealed result was opened before E was chosen.

## Method

1. **Sampling.** Sample v1.0 (Q8_0, llama.cpp) on the 832 train harmful prompts.
   - Thinking off: 4 samples with no system prompt, plus 4 with an "answer fully" system prompt
     (`scripts/v11b/sys_uncensored.txt`).
   - e0.9: 1 + 1.
   - Temperature 0.8, top-p 0.95. 8,320 samples in total.
2. **Judging.** Every sample is scored by Jev (p_fulfils). The system-prompt samples are much better: 85% useful vs
   53% without it.
3. **Answer caps.** Samples whose answer hit the original 768-token sampling cap (1,695 of them) were **continued** to
   8192 answer tokens (`cont_samples.py`) and re-judged. This is the same as having sampled with the large cap.
4. **SFT set.**
   - Per prompt and mode, keep the sample with the **highest p_fulfils**, if p ≥ 0.7. Exact ties go to shorter
     thinking.
   - Drop samples whose thinking hit its cap, refusal-phrase hits, and system-prompt samples whose thinking mentions
     the system prompt.
   - Train them **without** the system prompt (context distillation).
   - Add the original model's greedy answers on 1,561 harmless train prompts as anchors.
   - E's set: 1,186 harmful rows (off 691, e0.9 495) + 1,561 anchors = 2,747 rows.
5. **Training.** LoRA on the v1.0 BF16 checkpoint.
   - Targets: q, k, v, o, attention gate. Rank 16, alpha 32, lr 1e-4 cosine.
   - 2 epochs (687 steps), max-len 8192 (0 rows truncated), loss on completion tokens only.
   - Peak memory 91.5 GB on an RTX PRO 6000.
6. **Packaging.**
   - Merge in fp64 into the 120 BF16 attention tensors (mean relative delta 1.07%).
   - Patch the ggml-org BF16 GGUF with byte-exact name mapping; the patch script proves original bytes = base GGUF
     bytes first.
   - `llama-quantize Q8_0`.
   - Splits are seeded. Train, val and test are disjoint, and the test set stayed sealed until the finalist run.

Candidate lineage:

| candidate | data | epochs |
|---|---|---|
| A0 | off only | 1 |
| B | off + e0.9 | 1 |
| C | B data | 2 |
| D | top-2 per prompt | 1 |
| **E** | B recipe on 8192-continued data | 2 |

## Caps (applied to every model)

- Usefulness scoring and every number in this report use an **answer cap of 8192** for the original, v1.0,
  AtomicChat and all candidates. Think caps are unchanged at 4096.
  - At the old 512 cap, 23% (off) / 37% (e0.9) of v1.0's answers were cut mid-answer.
  - At 8192 nothing is above the 5% threshold: max 3.2% (E e0.9), all others ≤ 0.7%. So no subset re-scoring was
    needed.
  - `llama_eval.py` now reads `CMP_ANS_CAP` (default 512, so the compare bench is unchanged unless it is set).
  - The 8192 runs used ctx 49152 / np 6 instead of 40960 / 8 so long answers fit. With greedy decoding this does not
    change outputs, except for batch-level nondeterminism.
- **Training bias check.**
  - The first sampling pass used an answer cap of 768, and the SFT builder dropped unfinished rows. That excluded
    1,695 harmful samples, about 30% of the system-prompt samples. They were as good as the finished ones (p ≥ 0.7
    for 66–67% vs 67–71%).
  - They were continued to 8192 and re-judged. In E's set, long answers now win their prompt more often (E has 1,186
    harmful rows vs B/C's 919).
  - E's answers are longer than v1.0's and score higher at the 8192 cap, so there is no sign of a length penalty.
  - Training max-len is 8192, so no row was truncated.

## Known issues (for the release card and the fix round)

1. **Drift.** KLD 0.0318 vs v1.0's 0.0102. Wikitext PPL +4.7%, top-1 95.6%. Capability stayed inside the noise
   (MMLU-300 76.0, GSM8K-100 99), but the full capability suite has **not** been run on E.
2. **Over-thinking at e0.9.** 17% of harmful test prompts hit the 4096 thinking cap (v1.0: 13%). Some are long
   deliberations, or loops, on hard technical prompts.
3. **Soft refusals.** 1/277 off and 3/277 e0.9 by phrase. By hand, 2 are real soft refusals (a deflection on a
   weapon-making request; declining to argue for mixing bleach and ammonia), 1 is copyright-style hedging, and 1
   complies.
4. **Judge.** The same judge (Jev) picked the training data and scores the results. Checks against judge-gaming:
   - On val, C's dodge-marker rate (satire / "hypothetical" / disclaimers) fell from 18.1% to 12.8%.
   - A hand read of flipped cases showed real changes: v1.0 framed answers as "conceptual" or "debatable" or
     undercut them, while E/C did the task.
   - On the sealed test answers, E's dodge-marker rate fell from 15.9% to 11.2% (thinking off) and from 17.7% to
     14.4% (thinking on) vs v1.0, and median answers got about a third longer.
   - A second independent judge was not run.
5. **Not run:** the HF KL harness (llama.cpp KLD is reported instead), crisis and
   honest-limit probes (not run), and the full capability suite.

## Next round

1. **Over-thinking.** Rebuild SFT without runaway or looping thinking samples. Add short-thinking harmless e0.9
   anchors. Keep highest-p selection; never rank by answer length.
2. **Soft refusals.** Add targeted rows for "deflect a technical request" and "argue a dangerous claim".
3. **Drift.** Sweep the merge scale (0.6–0.9) on E's LoRA (saved) and trade usefulness against KLD.
4. **Capability suite** (the 6 benchmarks v1.0 passed) plus a second judge on a 50-prompt subset.

