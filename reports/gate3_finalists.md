# Gate 3: finalists (2026-11-21 15:28Z)

Selection rule (owner-approved, judge gate not met → option 4): phrase-list refusal ≤ 1% at thinking off and effort 0.9; honest limits 100% (hand-checked; the regex was widened); over-refusal ≤ 2%; MMLU-300 ≥ orig − 3; GSM8K-100 ≥ orig − 5; harmful e0.9 cap-hit ≤ OU's 19.6%; then lowest KL. Crisis is reported only (owner, 15:25Z). 2 finalists (owner, 15:05Z).

| | A: c1h_svd1_s2 | B: c1t013 | OU | orig |
|---|---|---|---|---|
| Method | C1: rank-2 refusal direction, svd1 partial conv-leak fix, constant s=2 on L4–23, row-norm preserving | C1: rank-2, svd1, kernel peaking L5–6 (s≈1.9, floor 0.88), row-norm preserving | W−sQQᵀW, s 2.5/4 | — |
| Refusal (off / e0.9) | 0.36% / 0% | 0% / 0.36% | 0% / 0.71% | 97.9% / 97.9% |
| KL / top-1 | 0.0395 / 93.3% | 0.0394 / 93.3% | 0.0637 / 91.6% | 0.0025 / 98.2% (noise) |
| Honest limits (hand-checked) | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| Crisis resources (reported only) | 90% / 80% | 90% / 80% | 100% / 90% | 100% / 100% |
| Over-refusal (borderline-200) | 0% | 0% | 0% | 36.5% |
| Harmful e0.9 cap-hit (1,536 cap) | 14.6% | 11.7% | 19.6% | 0% |
| MMLU-300 / GSM8K-100 | 74.3 / 99 | 75.3 / 97 | 75.0 / 97 | 77.0 / 98 |
| Edited tensors | attn.wo_ud L4–23 | attn.wo_ud L4–23 | attn.wo_ud L4–23 | — |

Why these two: they are the only full-pass candidates meeting every hard rule. Their KL is 38% below OU's. They tie on every metric within noise. A gets the full slow-benchmark suite (started earlier); B gets seeded subsets.
Not chosen: C3 (best c3t009, 2.1%/1.1% refusal > 1%); C2 (dominated, dropped); router stacks C4 (refusal gains, but KL ≥ 0.035 with no better full-pass point); "OU done right" (KL 0.134).
Missed targets: KL ≤ 0.025 (finalists are at 0.0395); top-1 ≥ 96% (93.3%). The hygiene metric is being re-validated (new loop detector) before it is applied.
