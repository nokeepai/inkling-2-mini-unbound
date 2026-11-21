# C3 (loop-aware ARA): Pareto, real val numbers (15:05Z)

## Ranking under the owner's 15:00Z rules
The rules, applied in order: phrase refusal ≤ 1% at both off and e0.9, honest = 100%, over-refusal ≤ 2%, MMLU ≥ orig−3, GSM ≥ orig−5, harmful e0.9 cap-hit ≤ OU's 19.6%, then lowest KL.

**No C3 point meets the refusal rule.** The two closest:
1. **c3t009 (best C3).** Refusal 2.1% off / 1.1% e0.9: fails ≤ 1%, though it is the nearest.
   - Passes everything else: honest 1.0, over-refusal 0.0%, MMLU 74.3 (≥ 74), GSM8K 98, cap-hit 15.0% (≤ 19.6%).
   - KL 0.0366, top1 93.7%.
2. **c3x001.** Refusal 2.85% / 2.85%. Also fails honest (0.9) and cap-hit (23.8% > 19.6%).
   - MMLU 76.7, GSM8K 97, over-refusal 0.5%. KL 0.0304.

The mpoa points that meet ≤ 1% (c1t013, c1h_svd1_s2, c1h_plain_s2, c1t017, at KL 0.039-0.045) outrank both. My recommendation is not to spend a finalist slot on C3. If one is needed, take c3t009.

**No thinking-preserve trials.** I did not fit the extra trials with an e0.9 harmful-thinking preserve set. A full eval takes about 16 min, so none could finish before the 15:15Z end.

Sources: results/runs.jsonl. KL is the kl_worker (HF, n=200 harmless val). The screen is fleet SGLang, phrase scorer, thinking off, n=120. The full stage is n=281 per mode. Judge rows are not used (judge gate).
Not yet back from the fleet: c3t015-c3t021 screens (all have val KL 0.008-0.019, at the c3t010 operating point; expect high refusal).

## Pareto front (val refusal vs val KL)
| cand | config | KL | top1 | refusal off / e0.9 | MMLU | GSM8K | e0.9 rep / cap | honest | over-ref |
|---|---|---|---|---|---|---|---|---|---|
| **c3x001** | L4-23, k16, pw 10, pres2 30, mean 1, γ 0.5 | **0.0304** | 93.9% | **2.85% / 2.85%** (full) | 76.7 | 97 | 54% / 24% ✗ | 0.9 ✗ | 0.5% |
| c3t009 | L7-23, k16, pw 17.8, lr 1.1e-4, 126 steps, mean 1.3 | 0.0367 | 93.7% | 2.1% / 1.1% (full) | 74.3 | 98 | 38% / 15% ✗ | 1.0 | 0.0% |
| c3x002 | L11-23, k16, pw 10, pres2 30, mean 2, γ 1, lr 5e-4 | 0.0263 | 94.1% | 9.2% (screen) | 76.0 | | | | |
| c3t011 | L11-23, k16, pw 10, pres2 10, mean 1, γ 0.5 | 0.0187 | 95.1% | 27.5% (screen) | 76.3 | | | | |
| c3t010 | L11-23, k16, pw 10, pres2 100, mean 1, γ 0.5 | 0.0128 | 96.1% | 54% (screen) | 76.0 | | | | |
| c3t014 | same, pres2 300, mean 0.5, γ 0.3 | 0.0078 | 97.0% | 72.5% (screen) | 77.0 | | | | |

- **Dominated:** 0% screen refusal only at KL 0.11-0.14 (c3t001/004 L4-23 r32, c3t007/008, c3t003). Also c3t000 (XQ-like, 0.046 / 38%), c3t005 (0.022 / 44%), c3t013 (0.0116 / 64%) and c3t012 (0.060 / 1.7%).
- **Best candidate: c3x001.** It fails the hard constraints: repetitive thinking at e0.9 is 54% (limit 3%), cap hits are 24% (limit 3%), and honest limits are 0.9 (needs 1.0).
- **c3t009** is the only c3 point with honest = 1.0, but it also fails rep / cap.
- **Not an Optuna trial:** c3x001/x002 are pod jobs from the proxy sweep (r3t001/r3t002), promoted by hand (scripts/ara/promote_job.py). build_cand is bit-exact (verify.json).

## vs mpoa (C1/C2), same workers
- **Low-refusal points:**
  - c1t013: KL 0.0394, 0% / 0.4%, rep 25%
  - c1h_svd1_s2: 0.0395, 0.4% / 0%, rep 27%
  - c1h_r1_svd1_s3_L11: 0.0353, 5.0% / 1.4%, rep 37%
  - c1h_r1_plain_s3_L11: 0.0382, 0.4% / 1.1%, rep 32%
- **C3 gives about 0.009 lower KL at a few % refusal** (c3x001 0.030 at 2.85%). mpoa reaches about 0% refusal, which C3 has not done below KL 0.1. C3's thinking hygiene is worse (rep 38-54% vs mpoa 25-37%).
- **Low-KL region:** C3 and C1 behave the same. Below KL 0.02, both families keep 27-92% refusal. c3t011 0.019 / 27.5% sits just ahead of C1's c1t010 0.029 / 11.7% and c1t001 0.018 / 37%.
- **Nobody passes the e0.9 repetition / cap constraints.** OU and c0 are at rep 27-34% too, and orig is at 0-2%. This looks like a property of removing refusal on this model, not of a single family.

## What we learned
1. **Removing refusal needs the early layers.** Restricted to L11-23, ARA brings the L23 refusal projection to the harmless level (proxy r47 ≈ 0) yet still refuses 54-72%. Refusal only goes when the edit reaches L4-10 (c3x001, c3t009 L7) or overshoots hard (γ ≥ 1, r47 ≈ -0.3 to -0.5). The XQ card's "L11-23 suffices" does not hold here.
2. **The broad preserve term is the KL lever.** It covers 32 thinking/reply tokens on 384 held-out harmless prompts, all passes. At L4-23 it cut KL from 0.108 (c3t001, refusal 0%) to 0.030 (c3x001, 2.85%).
3. **Proxies:** the KL proxy tracks the kl_worker (×1.0-1.4). The L23 refusal projection does not predict refusal rate. The pass-3 L17 projection (r41) does better: ≤ 0.2 gave ≤ 3% refusal.
4. **Next if C3 continues:** L4-23 with pres2 100-300 and γ 0.3-0.5 for lower KL. Add an e0.9 thinking-preserve set (harmful prompts' own non-refusal thinking tokens) to attack repetition. Both need a research GPU.
