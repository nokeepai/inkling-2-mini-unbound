# Inkling-2-Mini-Unbound v3: overnight build report (2026-11-22/23)

## 0. v3.1 update (2026-11-23, supersedes the section 1 recommendation)

**Recommendation: v3.1 candidate C beats v2. Ship it with thinking on, temperature 0.6, top_p 0.95 as the documented default.**

The owner dropped the KLD ≤ 0.04 gate ("make the best model, not optimize random numbers"). With that gate gone, the saved LoRAs were merged at full strength. No new training was needed.

| candidate | stack | KLD | verdict |
|---|---|---|---|
| **C** | sft1:1.0 + dpo2:1.0 | 0.0439 | **winner** |
| B | sft1:1.0 | 0.0474 | thinking-off gains like C, but e0.9 loops +7 to +11; 7.6–8.8 points worse than C on new e0.9 |
| A | sft1:0.6 + dpo2:1.0 | 0.0372 | ≈ C with thinking off, slightly worse at e0.9; eval stopped early by owner choice |

**C minus v2, paired, 95% CI.** Significant results are in bold.
- New test e0.9 useful: **Jev +6.8 [+2.1, +11.9]**, **Bison +5.2 [+1.3, +9.5]**.
- Old test off useful: **Bison +6.1 [+2.4, +10.2]**.
- Old test off held back: **Jev −8.4**, **Bison −6.5**.
- New test off useful: Jev +2.5 (n.s.); held back: **Jev −5.7**.
- Roleplay: no change.
- MMLU 74.3 [69.3, 79.3], GSM8K 98, over-refusal 1/784, old-test refusals 1/263 in each mode.
- Weak spot: at greedy e0.9, think-cap hits run **+5 to +6 points** above v2 (new test 23.4 → 29.8).

**Decoding fixes the loops.**
- **Test on v2**, e0.9, 240 rows. t0.6 cut cap hits 23.4 → 16.2 (significant) with no usefulness cost. DRY (0.8 / 1.75 / 2) did not cut loops and significantly raised held-back, with or without t0.6. Do not use DRY.
- **C at t0.6 against C greedy**: cap hits 29.6 → 15.9 (−13.7 [−19.7, −7.7]); Bison useful −2.2 (n.s.).
- **C at t0.6 against v2 greedy**: cap hits 22.6 → 16.1 (−6.5, significant); Bison useful 81.3 → 84.4 (+3.0, n.s.).
- The C t0.6 run was judged by Bison only: the Jev key (~/Code/jev/.env) disappeared from the Mac mid-session.

**Loop-fix DPO round on C: not done.** Mining finished on C: 1,049 greedy rows, 1,128 t0.8 resamples and 284 refused. But the training pod never finished building C's checkpoint. Spend had passed the $90 stop line, so both pods were deleted, and the mined data was lost with them. The t0.6 default already brings loops below v2. The round would mostly help greedy users.

**Artifacts.**
- The C GGUF is **not saved**: the private HF storage is full.
- Rebuild it from flashvenom/v11b-scratch `v3/loras/{sft1,dpo2}`: run `merge_stack.py` with `sft1:1.0 dpo2:1.0`, then `build_gguf.sh`.
- Eval outputs are in ~/.inkling-abl/v3_private/ev/{v31C, v31B, v31A, v31C_t06, decode}.

**v3.1 spend.**
- Pods: ink-v31-eval $31.24 and ink-v31-train $5.46, from `lium ps` at deletion.
- **Total** with the overnight run: **≈ $95.1 of $100**. That is past the $90 stop line, the result of an idle eval pod after mining finished at 16:11Z.
- Both pods are deleted, and the 18:00Z safety net fired with nothing left to delete.

## 1. Bottom line (overnight, under the KLD ≤ 0.04 gate; superseded by section 0)

**Recommendation: keep v2 as the release, and do not ship v3.**

The best v3 candidate is `dpo2_s0.5`: v2 plus SFT at merge scale 0.6, plus DPO at scale 0.5.
- It passes drift (KLD 0.0382), capability (MMLU and GSM8K within the original's CI) and over-refusal (0.13%).
- It is **not a clear improvement over v2**. Of 25 paired comparisons, one is significantly better: old test, thinking off, Bison useful +3.8 points [+0.8, +7.2].
- The stated loop goal moved the wrong way. With thinking on (e0.9), think-cap hits are **+5–6 points worse** than v2 (old test +5.9 [+0.0, +11.9]; new test +5.2 [−1.3, +11.2]).
- No candidate, v2 included, meets the useful ≥90%, roleplay held-back ≤7% or e0.9 cap-hit ≤10% gates on these test sets.

Why it fell short, in one sentence: **the KLD ≤ 0.04 gate leaves v3 only ~0.008 of drift headroom over v2 (0.0318)**, and at that budget SFT and DPO moved the behaviour too little to measure. Section 8 lists what to try next.

## 2. What v3 is

A drop-in for v2: the same 120 BF16 attention tensors (q, k, v, o and gate across 24 layers), with the same names, shapes and layout. There are no router or expert edits; methods/experts/C4_FEASIBILITY.md says those failed. It is built in two stages on top of v2:

1. **SFT (on-policy rejection sampling + context distillation).**
   - v2 Q4_K_M sampled training prompts at effort 0 and 0.9, with and without an "answer fully" system prompt. That is 3,603 harm-train samples plus 1,467 pool samples at build time (creative, prefill, harm-pool and roleplay).
   - Jev (TypeSafe) scored every sample. The best sample per prompt and mode was kept: p_fulfils ≥ 0.7, not held back, shortest thinking within a 0.05 p-bin.
   - Samples were dropped for:

     | reason | samples |
     |---|---|
     | held back | 1,760 |
     | cap hit or unfinished | 582 |
     | thinking cites the system prompt | 383 |
     | roleplay flags | 210 |
     | compliance deliberation (narrowed filter, ~3%) | 155 |
     | refusal or hedge | 75 |
     | slop | 25 |
     | loop | 6 |

   - The "answer fully" system prompt was removed from the training text (context distillation). Per-row creative and roleplay systems were kept.
   - The SFT set had **1,822 rows**: 1,089 harm and creative, 33 roleplay, and 700 original-model harmless anchors. 18 over-long rows were dropped.
   - LoRA r16/α32 on attention, lr 5e-5, 1 epoch, 226 steps.
2. **DPO on top of SFT (at scale 0.6).** 409 pairs: 240 *full*, 114 *loop*, 51 *loopthink* and 13 *opener*; 9 over-long pairs dropped. The pair kinds:
   - *full*: a selected answer against a held-back or refusing no-system sample. DPO on answer tokens only.
   - *opener*: the first 32 answer tokens after shared thinking.
   - *loop*: a finished, shorter e0.9 trace against a greedy cap-hit or looping trace. DPO on the whole sequence.
   - *loopthink*: like *loop*, but DPO on thinking tokens only, because the finished trace's answer was held back.
   - Memory-exact two-pass DPO, LoRA r32, β 0.1, **lr 3e-5**, 52 steps. Reward accuracy rose from 0.5 to ~0.75–0.8, and z from 0 to ~0.6–1.0.
   - A first DPO run at lr 5e-6 learned nothing (z ≈ 0 for all 63 steps) and was discarded.
3. **Loop-pair mining** for (2):
   - v2 ran greedy at e0.9 on 1,200 non-test prompts (600 pool, 600 harmless/technical). Cap-hit or looping rates by prompt type: **creative 70%**, harm-pool 20%, **harmless 6.5%**.
   - Those prompts were resampled at temperature 0.8 (n = 3), giving 147 loop pairs.
4. **Merge**: W = bf16(W_v2 + 0.6·ΔSFT + s·ΔDPO), computed in fp64 with a single rounding. The result is patched into the ggml-org BF16 GGUF with a byte-verified tensor mapping, then quantized with llama-quantize Q8_0.

Bison (OpenRouter baseten/bison-1) is used **only for grading, never for selection**, so it is the judge not trained against.

## 3. Owner rules applied

- **Mass-casualty weapons content excluded.** Every prompt labelled with the biological, chemical, nuclear/radiological weapons category (112 ids), plus 54 ids excluded earlier, was removed. This covers training prompts, sampling pools, candidates, the sealed test sets (the old sealed test went from 277 to 263 prompts) and all filter work. The new v3 pool and test were screened for this at build time. Excluded prompts were never inspected, rewritten, printed or analysed. All selection and analysis ran on counts and scores only; no sample text was printed.
- **No truncated training targets.** Rows longer than the training window (4096 tokens) were **dropped, never cut**: 18 of 1,822 SFT rows and 9 of 418 DPO pairs. The only exception is the *rejected* side of a loop pair, which is a runaway thinking trace that is only pushed down, never imitated (162 pairs). Gradient checkpointing and micro-batch 1 were already on; 8192 tokens per sequence runs out of memory on the 96 GB card (73 GB at 4096).
- **No publishing.** The production pods, app and API were not touched.

## 4. Evaluation setup

- llama.cpp @19bc41dbb, Q8_0, greedy, think cap 4096, answer cap 8192.
- **New sealed v3 test**:
  - 480 single-turn prompts (360 harmful + 120 creative), all run with thinking off.
  - At e0.9: the 240 even rows for candidates. The v2 file has 272 rows; comparisons are paired by id.
  - 30 roleplay chats × 8 turns.
- **Old sealed test** (v2 report): 263 harmful prompts, borderline over-refusal set (784), MMLU-300, GSM8K-100.
- **Drift**: wikitext KLD against the ggml-org original Q8_0 (40 chunks × 4096 tokens). v2 was re-measured on this setup and matches its release figure.
- **Judges**: Jev useful = p_fulfils ≥ 0.5 (the selection judge), and Bison useful (independent).
- **Statistics**: paired bootstrap 95% CIs against v2 on the same prompts.
- **Gaps**:
  - dpo2's old-test e0.9 run finished 219 of 263 prompts before the eval pod hit its budget stop.
  - A few rows per file lack a judgment (n is shown in each cell).

## 5. Results

### 5.1 Gates (best v3 candidate = dpo2_s0.5)

| gate | target | original | v1 | **v2** | sft1_s0.6 | **v3 = dpo2_s0.5** | best competitor (AtomicChat) |
|---|---|---|---|---|---|---|---|
| useful, thinking off (old test, Jev / Bison) | ≥ 90% | 0.8 / 0.8 | 51.3 / 73.8 | 79.8 / 87.1 | 82.1 / 88.6 | **83.7 / 90.9** | 52.1 / 69.2 |
| useful, thinking on (old test, Jev / Bison) | ≥ 90% | 0.8 / 1.1 | 61.6 / 76.8 | 82.1 / 90.1 | 85.9 / 93.5 | **82.2 / 91.3** (n=219) | 55.5 / 71.5 |
| useful, new test off / e0.9 (Jev) | ≥ 90% | — | — | 75.8 / 71.7 | 73.7 / 70.6 | 77.5 / 69.4 | — |
| refusal phrase, off / on (old test) | ≤ 0.3% | 96.2 / 95.8 | 0.0 / 0.0 | 0.38 / 1.14 | 0.00 / 0.76 | 1.14 / 0.00 | 0.0 / 0.0 |
| refusal phrase, new test off / e0.9 | ≤ 0.3% | — | — | 0.63 / 1.10 | 0.83 / 0.42 | 0.21 / 0.42 | — |
| roleplay held back (Jev / Bison) | ≤ 7% | — | — | 60.4 / 31.2 | 56.7 / 30.4 | 61.3 / 28.3 | — |
| e0.9 think-cap hits (old / new) | ≤ 10%, no loops | 0.4 / — | 11.0 / — | 17.1 / 21.0 | 24.7 / 26.7 | **23.3 / 28.3** | 8.0 / — |
| over-refusal (borderline 784) | ≤ 0.5% | 38.0 | 0.0 | 0.3 | 0.13 | **0.13** | — |
| MMLU-300 / GSM8K-100 | within original CI (76.3 / 97) | 76.3 / 97 | 75.7 / 99 | 76.0 / 99 | 74.7 / 97 | **74.3 [69.7, 79.0] / 98** | — |
| KLD vs original / same top token | ≤ 0.04 | — | 0.0102 | 0.0318 / 95.6% | 0.0397 / 95.5% | **0.0382 / 95.5%** | — |

Notes:
- Original, v1 and AtomicChat old-test numbers come from the v2-run outputs restricted to the same 263 ids. Their Jev scores came from an earlier judging pass, so cross-model Jev differences of about 2 points are within judging noise. The v2 and v3 columns were re-judged with the same judge_v3 pass.
- The roleplay judges disagree strongly in level (Jev "held" is about twice Bison's). The gate of ≤ 7% is far from all models on this 30-chat test.
- The new test is harder than the old one: v2 scores about 4 points lower on it.

### 5.2 Paired comparison: v3 (dpo2_s0.5) minus v2

Each cell is v2 → v3 (difference [95% CI], n). In the roleplay row, the first two columns show **refused** rather than useful.

| set | Jev useful | Bison useful | Jev held | Bison held | think-cap hits |
|---|---|---|---|---|---|
| new test, off | 76.3 → 77.5 (+1.3 [−1.9, +4.5], 472) | 85.5 → 85.3 (−0.2 [−3.2, +2.8], 468) | 44.5 → 41.5 (−3.0 [−7.0, +0.9]) | 37.2 → 36.1 (−1.1 [−4.9, +2.8]) | 4.2 → 4.9 (+0.6 [−1.5, +2.7]) |
| new test, e0.9 | 72.0 → 69.4 (−2.6 [−6.9, +1.7], 232) | 81.5 → 79.7 (−1.7 [−6.0, +2.6]) | 40.5 → 41.8 (+1.3 [−4.7, +7.3]) | 40.1 → 38.4 (−1.7 [−7.3, +4.3]) | 22.4 → 27.6 (+5.2 [−1.3, +11.2]) |
| roleplay (refused) | 52.1 → 53.8 (+1.7 [−4.6, +7.9], 240) | 20.4 → 16.7 (−3.8 [−9.2, +1.7]) | 60.4 → 61.3 (+0.8 [−5.4, +7.1]) | 31.2 → 28.3 (−2.9 [−9.2, +3.3]) | 7.1 → 7.5 (+0.4 [−3.8, +4.6]) |
| old test, off | 79.8 → 83.7 (+3.8 [−0.4, +8.0], 263) | **87.1 → 90.9 (+3.8 [+0.8, +7.2])** | 53.6 → 49.0 (−4.6 [−9.9, +0.8]) | 42.6 → 38.8 (−3.8 [−9.1, +1.9]) | 4.2 → 3.0 (−1.1 [−4.2, +1.9]) |
| old test, e0.9 | 81.7 → 82.2 (+0.5 [−4.1, +5.5], 219) | 90.0 → 91.3 (+1.4 [−2.7, +5.5]) | 51.1 → 51.6 (+0.5 [−6.4, +6.9]) | 39.7 → 43.4 (+3.6 [−2.3, +10.1]) | **17.4 → 23.3 (+5.9 [+0.0, +11.9])** |

SFT-only (sft1_s0.6) against v2 shows the same pattern:
- Useful moves −2 to +4, and none of those changes is significant.
- Old-test e0.9 think-cap hits are worse: +7.6 [+2.3, +12.9].
- New-test thinking-off Jev held drops −4.0 [−7.8, −0.2].

Full table in `~/.inkling-abl/v3_private/ev/paired_sft1.md`.

### 5.3 Drift sweep (wikitext KLD against the original; v2 = 0.0318)

| candidate | KLD | same top | PPL ratio |
|---|---|---|---|
| pilot (617-row SFT, scale 1.0) | 0.0422 | 95.46% | 1.063 |
| sft1 scale 1.0 | 0.0474 | 95.35% | — |
| **sft1 scale 0.6** | 0.0397 | 95.46% | 1.059 |
| sft1 0.6 + dpo2 0.5 | **0.0382** | 95.49% | 1.056 |
| sft1 0.6 + dpo2 1.0 | 0.0372 | 95.48% | — |

DPO *lowered* drift. The DPO2-at-1.0 variant passes KLD but was not behaviour-evaluated: there was no time left before the eval pod's budget stop.

## 6. Artifacts

- **GGUF: not saved.** The private upload of the Q8_0 failed: *"Private storage limit reached for user flashvenom"*. I did not delete anything from the scratch repo to make room; that is the owner's call. The candidate GGUFs were lost when the eval pod was deleted.
- **What was saved privately** (flashvenom/v11b-scratch, `v3/loras/`): `sft1.pt` (35 MB) and `dpo2.pt` (70 MB). Local copies are in `~/.inkling-abl/v3_private/loras/`.
- **Rebuilding the GGUF exactly** takes one pod and ~10 min, with deterministic fp64 merge, byte-verified patch and llama-quantize:
  ```
  python3 merge_stack.py /data/v2hf hf_name_map.json out.safetensors sft1.pt:0.6 dpo2.pt:0.5
  bash build_gguf.sh v3 out.safetensors
  ```
  The dpo2 scale is 0.5; use 1.0 for the untested variant.
- **Side-by-side** v2 against dpo2_s0.5 on the new test and roleplay: `~/.inkling-abl/v3_private/side_by_side_v3.html`, local only, chmod 600. Rows where v3 gained come first, then rows where it lost.
- **Data and evals** (private, local): `~/.inkling-abl/v3_private/`, with build1/, mine/ and ev/{v2, pilot, sft1_s0.6, dpo2_s0.5}/ holding metrics, judgments and paired tables.

## 7. Pods and spend

| pod | GPUs | $/h | role | benchmark | cost |
|---|---|---|---|---|---|
| ink-v3-eval | 6× RTX 5090 | 3.90 | R1 sampling, all candidate evals, GGUF builds, KLD | 384 CPU, 755 GB RAM, disk 2.6–3.1 GB/s, HF down ~100 MB/s, HF up ~43 MB/s with hf_transfer (a raw upload test misleadingly said 0.6 MB/s) | $28.28 (hit its budget stop 04:29Z) |
| ink-v3-gen2 | 8× RTX 5090 | 5.20 | v2 baseline eval + pool sampling; deleted early at 23:50Z | not benchmarked (first build failed on the tarball download) | ~$11.6 |
| ink-v3-train | 1× RTX PRO 6000 96 GB | 1.70 | pilot, SFT1, DPO1, DPO2 | 8 CPU, 62 GB RAM, disk ~135 MB/s, HF ~100 MB/s | ~$10.4 |
| ink-v3-mine | 8× RTX 5090 (HK) | 5.60 | loop-pair mining, 01:38–02:33Z | setup to serving ~15 min | ~$5.5 |

- **Pods** ≈ **$55.8**.
- **API** ≈ **$2.6**: Venice teacher $0.70, v3 data build $0.76, Bison $1.10.
- Jev (TypeSafe) reported $0.00, and its billing is not visible.
- **Total ≈ $58.4** of the $100 cap.
- All four pods are deleted, the Mac watchers are stopped and no TTL was ever extended.

## 8. What went wrong, and what to try next

**What went wrong (and the time each cost):**
- *SFT1 ran out of memory* at max-len 8192. The waiter only checked for the output file, so it took ~50 min to notice. Waiters now also check that the process is alive.
- *DPO1 at lr 5e-6 learned nothing* (~35 min). LoRA DPO here needs about 3e-5.
- *Half the loop mining was lost to a missing field* (harmless prompts). It was redone on an extra pod.
- *KLD ran out of GPU memory twice*, because a previous eval's llama-servers still held GPU 0.
- *HF private storage was full*, so the final GGUF could not be saved.

**Why the gains are small:**
1. **Drift budget.** v2 already uses 0.0318 of the 0.04 KLD budget. Every v3 change had to be scaled down to fit (SFT at 0.6), and the remaining delta was too small to move behaviour measurably.
2. **Loops got worse after SFT.** The SFT set's e0.9 rows had bounded thinking (median 795, p90 1,835 tokens). Even so, greedy e0.9 cap hits rose by 5–8 points. The 147 loop pairs in DPO2 did not undo this at scale 0.5.

**Next steps if the owner wants another round:**
- **Relax or re-base the KLD gate.** For example, measure drift against v2 rather than the original, or allow ≤ 0.045. This is the single biggest lever, and it is the owner's call.
- **Treat loops as a decoding problem first.** Greedy e0.9 loops on 70% of creative prompts in v2. A repetition-aware sampler default, or a recommended non-greedy setting, might do more than training.
- **If training,** mine many more loop pairs, especially creative ones (most loops are there), and run loop-only DPO at full scale. Then measure KLD; DPO lowered drift here.
- Evaluate `sft1 0.6 + dpo2 1.0` (KLD 0.0372), which was never behaviour-tested.
