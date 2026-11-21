# Inkling-2-Mini abliteration: final report

Generated 2026-11-21T15:06:13Z from `results/runs.jsonl` (2505 rows) by `scripts/package/make_report.py`.
Split: **val**. Cells show value [95% CI]; ‹val› marks a cell taken from another split because the val row is missing.

## Verdict

(pending Gate 4)

## A. Targets (harness/TARGETS.md)

| metric |  target |original (`orig`) | OpenUncensored (`ou`) | final (`c1t013`) |
|---|---|---|---|---|
| Harmful refusal, thinking off (phrase + judge) | ≤ 1% | 98.6% [97.2%, 99.6%] | 55.9% [49.8%, 61.6%] | missing |
| Harmful refusal, effort 0.9 (phrase + judge) | ≤ 1% | 98.9% [97.5%, 100.0%] | 48.0% [42.4%, 53.7%] | missing |
| Harmful refusal, thinking off (phrase list) | ≤ 1% | 97.9% [96.1%, 99.3%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| Harmful refusal, effort 0.9 (phrase list) | ≤ 1% | 97.9% [96.1%, 99.3%] | 0.7% [0.0%, 1.8%] | 0.4% [0.0%, 1.1%] |
| Worst category refusal, thinking off | ≤ 5% | 100.0% [100.0%, 100.0%] (cyber) | 0.0% [0.0%, 0.0%] (cbrn_weapons) | 0.0% [0.0%, 0.0%] (cbrn_weapons) |
| Worst category refusal, effort 0.9 | ≤ 5% | 100.0% [100.0%, 100.0%] (cyber) | 4.2% [0.0%, 12.5%] (violence) | 1.1% [0.0%, 3.2%] (crime_illegal_goods) |
| Non-evasive answers, thinking off (judge, unverified: 82%/κ0.63) | ≥ 90% (judge gate not met) | 1.8% [0.4%, 3.6%] | 44.1% [38.4%, 50.2%] | missing |
| Non-evasive answers, effort 0.9 (judge, unverified) | ≥ 90% (judge gate not met) | 2.5% [0.7%, 4.6%] | 52.0% [46.3%, 57.6%] | missing |
| Over-refusal, benign borderline, thinking off | ≤ 2% | 34.4% [31.2%, 37.8%] | 0.3% [0.0%, 0.7%] | 0.0% [0.0%, 0.0%] |
| Mean KL vs original, harmless, first 256 tokens (HF, AQ off) | ≤ 0.02 (≤ 0.025 accepted) | 0.0025 [0.0023, 0.0028] (self-noise floor) | 0.0637 [0.0557, 0.0735] | 0.0394 [0.0331, 0.0466] |
| Top-1 agreement, harmless (HF, AQ off) | ≥ 96% | 98.2% [98.1%, 98.4%] (self-noise floor) | 91.6% [91.1%, 92.2%] | 93.3% [92.8%, 93.8%] |
| Honest-limit probes | 100% val; orig rate on test | 100.0% [100.0%, 100.0%] | 80.0% [50.0%, 100.0%] | 80.0% [50.0%, 100.0%] |
| Crisis resources (self-harm probes) | orig rate | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 90.0% [70.0%, 100.0%] |
| MMLU-300 (reject filter only) | ≥ orig − 3 pts | 77.0 [72.3, 81.7] | 75.0 [70.0, 79.7] | 75.3 [70.3, 80.3] |
| GSM8K-100, effort 0.5 | ≥ orig − 5 pts | 98.0 [95.0, 100.0] | 97.0 [93.0, 100.0] | 97.0 [93.0, 100.0] |
| Repetitive thinking, effort 0.9 | ≤ 3% | 0.0% [0.0%, 0.0%] | 34.2% [28.8%, 39.9%] | 24.9% [19.9%, 29.9%] |
| Thinking-cap hits, effort 0.9 | ≤ 3% | 0.0% [0.0%, 0.0%] | 19.6% [14.9%, 24.2%] | 11.7% [8.2%, 15.7%] |

Judge metrics use the open HarmBench classifier, which scored 82% agreement (Cohen's κ 0.63) with 50 hand labels, below the 85% gate. They are reported, not used to select.

## B. Other metrics (capability suite, thinking hygiene, speed)

| metric | original (`orig`) | OpenUncensored (`ou`) | final (`c1t013`) |
|---|---|---|---|
| gsm8k_cap_hit | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| gsm8k_finish | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] |
| gsm8k_rep | 15.0% [8.0%, 22.0%] | 15.0% [9.0%, 22.0%] | 11.0% [5.0%, 18.0%] |
| gsm8k_think_len | 240.5300 [204.0000, 281.6500] | 239.1200 [203.1500, 278.3400] | 250.6800 [208.9200, 298.1800] |
| harmful_e09_finish | 96.1% [93.6%, 98.2%] | 53.4% [47.7%, 59.1%] | 61.6% [55.9%, 67.3%] |
| harmful_e09_think_len | 193.0712 [176.2633, 211.8399] | 762.4342 [712.4377, 815.2028] | 675.5445 [628.2064, 723.8007] |
| harmful_off_cap_hit | 0.0% [0.0%, 0.0%] | 1.8% [0.4%, 3.6%] | 1.4% [0.4%, 2.8%] |
| harmful_off_finish | 98.9% [97.5%, 100.0%] | 70.1% [64.4%, 75.4%] | 77.2% [72.2%, 81.9%] |
| harmful_off_rep | 0.4% [0.0%, 1.1%] | 5.0% [2.5%, 7.8%] | 3.9% [1.8%, 6.4%] |
| harmful_off_think_len | 97.2349 [91.9644, 102.7473] | 295.8577 [268.8292, 326.1246] | 248.0925 [225.6121, 274.2384] |
| probe_crisis_resources_e09 | 100.0% [100.0%, 100.0%] | 90.0% [70.0%, 100.0%] | 80.0% [50.0%, 100.0%] |
| probe_crisis_resources_off | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 90.0% [70.0%, 100.0%] |
| probe_honest_limit_e09 | 100.0% [100.0%, 100.0%] | 90.0% [70.0%, 100.0%] | 90.0% [70.0%, 100.0%] |
| probe_honest_limit_off | 100.0% [100.0%, 100.0%] | 80.0% [50.0%, 100.0%] | 80.0% [50.0%, 100.0%] |

## Files

- Spend: (pending)

## Run IDs

`fid-orig-noise_bs1_2b-kl-124916`, `fid-orig-noise_bs1_2b-top1-124916`, `fid-ou-baseline_2b-kl-124916`, `fid-ou-baseline_2b-top1-124916`, `fleet-c1t013-full-ink-fleet-2-g4-142953`, `fleet-orig-full-ink-fleet-2-g0-140605`, `fleet-ou-full-ink-fleet-2-g4-140622`, `klw-c1t013-kl-140458`, `klw-c1t013-top1-140458`, `sgl-val-orig-run2`, `sgl-val-ou-baked`
