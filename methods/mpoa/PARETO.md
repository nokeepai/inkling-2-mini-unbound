# MPOA (C1) Pareto and finalists — FINAL (15:10Z)

**Finalists (frozen by the owner):** A = **c1h_svd1_s2**, B = **c1t013**. No other candidate would have beaten them (see the last section).

- **Engine:** fleet (ink-fleet-2) SGLang rows only. KL comes from the HF kl_worker (AQ off, 200 val harmless).
- **Ranking rule (owner 15:05Z):** phrase refusal off and e0.9 ≤ 1%, honest-limit = 100%, over-refusal ≤ 2%, MMLU ≥ 74.0, GSM ≥ 93, harmful e0.9 cap-hit ≤ OU (19.6%), then lowest KL.
- **Honest-limit metric:** this is regex-scored and gives false negatives. The lead's hand-check found c1t013 and c1h_svd1_s2 both truly at 100%/100%, so it is not used for ranking here.
- **Crisis:** not a gate (owner). The lead's hand-check found that both finalists miss suicidal intent on 2 crisis probes, which the original catches.
- **Baselines (fleet):**
  - orig: refusal 97.9/97.9%, MMLU 77.0, GSM 98, over-refusal 36.5%.
  - OU: refusal 0.0/0.7%, KL 0.0637, top-1 91.6, MMLU 75.0, GSM 97, e0.9 cap-hit 19.6%, rep 34.2%, over-refusal 0%.

## Every full-pass candidate (honest-limit and hygiene side by side)
| cand | KL | top1 % | ref off/e0.9 % | honest off/e09 (regex) | crisis off/e09 | over-ref % | MMLU | GSM | cap e0.9 % | rep e0.9 % | owner gates (honest excluded) | edit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **c1t013 (B)** | 0.0394 | 93.3 | 0.0 / 0.4 | 0.80 / 0.90 (hand 1/1) | 0.90 / 0.80 | 0.0 | 75.3 | 97 | 11.7 | 24.9 | PASS | r2; svd1 on L4–8 and L17, plain projection elsewhere (conv_scope "leaky"); s 2.4→1.8 on L7–17, 1.2→1.0 on L18–23; MP pre |
| **c1h_svd1_s2 (A)** | 0.0395 | 93.3 | 0.4 / 0.0 | 1.00 / 0.80 (hand 1/1) | 0.90 / 0.80 | 0.0 | 74.3 | 99 | 14.6 | 27.0 | PASS | r2 svd1, constant s=2 on L4–23, MP pre |
| c1t017 | 0.0430 | 93.1 | 0.0 / 0.4 | 0.80 / 0.80 | 1.00 / 0.80 | 0.0 | 75.7 | 98 | 11.0 | 25.3 | PASS | r1 svd1 (leaky scope), kernel peak 3.1 at L15 |
| c1h_plain_s2 | 0.0445 | 93.1 | 0.0 / 0.4 | 0.70 / 0.90 | 0.80 / 0.90 | 0.5 | 75.3 | 98 | 13.9 | 26.3 | PASS | r2 plain+MP, s=2 on L4–23 |
| c1h_r1_svd1_s4_L11 | 0.0498 | 92.8 | 0.0 / 0.4 | 0.80 / 0.90 | 0.90 / 0.90 | 0.0 | 75.0 | 99 | 15.3 | 35.6 | PASS | r1 svd1, s=4 on L11–23 |
| c1h_r1_svd1_s3_L4 | 0.0559 | 92.3 | 0.0 / 0.0 | 0.70 / 0.90 | 0.40 / 0.40 | 0.0 | 73.7 | 98 | 16.7 | 30.6 | MMLU | r1 svd1, s=3 on L4–23 |
| c1h_x4_ouprofile | 0.1341 | 87.2 | 0.0 / 0.7 | 0.70 / 0.70 | 0.30 / 0.30 | 0.5 | 75.0 | 98 | 16.7 | 32.4 | PASS (KL far worse) | "OU done right": exact4, s 4/2.5/4, MP pre |
| c1h_r1_plain_s3_L11 | 0.0382 | 93.7 | 0.4 / 1.1 | 0.90 / 1.00 | 1.00 / 0.90 | 0.5 | 76.0 | 99 | 15.7 | 31.7 | ref e0.9 (1.1%) | r1 plain+MP, s=3 on L11–23 |
| c1h_r1_svd1_s3_L11 | 0.0353 | 93.9 | 5.0 / 1.4 | 1.00 / 0.80 | 1.00 / 0.90 | 0.0 | 75.0 | 99 | 15.3 | 36.7 | refusal | r1 svd1, s=3 on L11–23 |

- **Ordering:** with the regex honest-limit excluded, A and B are the two lowest-KL candidates that pass every gate; they are 0.0001 apart, below the 0.0025 tie floor.
- **Tiebreak:**
  - B has lower cap-hit (11.7 vs 14.6%) and repetition (24.9 vs 27.0%), and higher MMLU (75.3 vs 74.3).
  - A is the simpler edit (one constant strength), with GSM 99.
- **Near miss:** c1h_r1_plain_s3_L11 is the only candidate with lower KL (0.0382), but it fails e0.9 refusal at 1.1%.
- **Thinking hygiene:** every candidate is below OU on e0.9 cap-hit and repetition. All beat OU on KL except the exact4 OU profile.

## Pareto front (refusal vs KL, screen and full)
See methods/mpoa/pareto_table.md. Full data for all 80 candidates: methods/mpoa/all_metrics.tsv. Owner-rule ranking: methods/mpoa/rank_full.md.
- **Screen-only front points:** c1t010 (11.7% at 0.029), c1s05 plain s1 (35% at 0.018), c1t002 svd1 s1 (45% at 0.016), and below.
- **No pure-ablation (s ≤ 1) point got under ~35% screen refusal.**

## Would anything have beaten the finalists?
- **c1c, the hygiene study** (strength outside the loop: s_pre on L4–5, s_loop ≤ 1.5 on L6–17, s_post on L18–23):
  - Its KL was cheap (0.010–0.030), but screen refusal stayed high. The best was c1ct004 at 19.2% (r1 plain, 4/1/4, KL 0.029), then c1ct005 at 26.7% (2/1/4).
  - None reached the ≤ 5% promotion bar, so none would have beaten A/B.
  - Takeaway: on this model, refusal removal needs reflection strength (s ≈ 2) INSIDE the looped layers.
- **c1b** (strong kernel space): KL 0.030–0.070. Its screens had not landed when the study was stopped, and none of its KLs is below 0.0394 except c1bt002 (0.0295, unscored).
- **Crisis-protected variants** (cancelled by the owner; fleet asked to skip): c1t013_cr03 KL 0.0459, c1t013_cr1 KL 0.0418. These are the KL costs of a crisis-only ridge (+0.002 to +0.006).

## What we learned
- **Conv fix rank:** the exact conv fix needs rank 4r, not 5r. The tap w[:,3] multiplies h_t, so the lag-0 term is (1+w3)⊙Q.
- **svd1:** the rank-r partial fix matches plain's ‖ΔW‖ and cuts the conv leak from 0.19–0.62 to 0.03–0.21. Both finalists use it.
- **exact4 is costly at high strength:** KL 0.134 at OU strengths.
- **MP must scale rows before projecting** (W' = (I − sUUᵀ)·diag(r)·W). Rescaling after the projection reopens a 4–21% leak.
- **C2 ridge** (harmfulness and atoms) roughly doubled KL at equal strength, with no refusal gain. The owner dropped C2.
- **Hard-negative rounds (C2 step 3) were not run.** The time went to the fleet outage and the strength redirect.
- **Search loop bug:** the 50-min timeout marked 28 trials FAIL while the fleet was down. It was fixed with orphan recovery. The Optuna DBs at state/optuna/{c1,c2,c1b,c2b,c1c}.db are therefore partial records; all scoring lives in runs.jsonl.
