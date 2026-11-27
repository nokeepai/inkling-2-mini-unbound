### per suite (repeat r0; 10,000 resamples, seed 1234)

| suite | metric | tag | n | score | 95% Wilson | Δ vs v21 (pts) | 95% paired bootstrap | McNemar p | ref-only/tag-only right | items | non-inferiority (lower ≥ −2 pts) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pub_humaneval | pass@1 | v21 | 164 | 98.2% | [94.8%, 99.4%] | ref |  |  |  |  |  |
| pub_humaneval | pass@1 | v3 | 164 | 98.8% | [95.7%, 99.7%] | +0.6 | [-1.2, +2.4] | 1 | 1/2 | match | non-inferior |
| pub_mbpp | pass@1 | v21 | 257 | 97.7% | [95.0%, 98.9%] | ref |  |  |  |  |  |
| pub_mbpp | pass@1 | v3 | 257 | 96.1% | [93.0%, 97.9%] | -1.6 | [-3.5, +0.0] | 0.219 | 5/1 | match | inconclusive |
| pub_humanevalfix | pass@1 | v21 | 164 | 95.7% | [91.5%, 97.9%] | ref |  |  |  |  |  |
| pub_humanevalfix | pass@1 | v3 | 164 | 96.3% | [92.2%, 98.3%] | +0.6 | [-2.4, +3.7] | 1 | 3/4 | match | inconclusive |
| pub_quixbugs | pass@1 | v21 | 31 | 96.8% | [83.8%, 99.4%] | ref |  |  |  |  |  |
| pub_quixbugs | pass@1 | v3 | 31 | 90.3% | [75.1%, 96.7%] | -6.5 | [-19.4, +6.5] | 0.625 | 3/1 | match | inconclusive |
| pub_gsm8k_full | strict | v21 | 1319 | 71.9% | [69.4%, 74.2%] | ref |  |  |  |  |  |
| pub_gsm8k_full | strict | v3 | 1319 | 88.3% | [86.5%, 89.9%] | +16.5 | [+14.3, +18.6] | 1.79e-52 | 13/230 | match | non-inferior, better |
| pub_gsm8k_full | number | v21 | 1319 | 96.7% | [95.6%, 97.5%] | ref |  |  |  |  |  |
| pub_gsm8k_full | number | v3 | 1319 | 96.7% | [95.6%, 97.6%] | +0.1 | [-0.6, +0.8] | 1 | 11/12 | match | non-inferior |
| pub_gsm8k | strict | v21 | 300 | 68.0% | [62.5%, 73.0%] | ref |  |  |  |  |  |
| pub_gsm8k | strict | v3 | 300 | 88.0% | [83.8%, 91.2%] | +20.0 | [+15.3, +25.0] | 1.3e-15 | 3/63 | match | non-inferior, better |
| pub_gsm8k | number | v21 | 300 | 97.3% | [94.8%, 98.6%] | ref |  |  |  |  |  |
| pub_gsm8k | number | v3 | 300 | 97.7% | [95.3%, 98.9%] | +0.3 | [-1.3, +2.0] | 1 | 3/4 | match | non-inferior |
| pub_math500 | is_equiv | v21 | 500 | 92.2% | [89.5%, 94.2%] | ref |  |  |  |  |  |
| pub_math500 | is_equiv | v3 | 500 | 93.2% | [90.6%, 95.1%] | +1.0 | [-0.6, +2.6] | 0.302 | 5/10 | match | non-inferior |
| pub_mmlu_2k | accuracy | v21 | 2000 | 87.6% | [86.1%, 89.0%] | ref |  |  |  |  |  |
| pub_mmlu_2k | accuracy | v3 | 2000 | 87.2% | [85.6%, 88.5%] | -0.5 | [-1.6, +0.5] | 0.403 | 63/53 | match | non-inferior |
| pub_mmlu | accuracy | v21 | 400 | 87.0% | [83.3%, 89.9%] | ref |  |  |  |  |  |
| pub_mmlu | accuracy | v3 | 400 | 87.0% | [83.3%, 89.9%] | +0.0 | [-2.0, +2.2] | 1 | 9/9 | match | non-inferior |
| code | pass@1 | v21 | 40 | 90.0% | [76.9%, 96.0%] | ref |  |  |  |  |  |
| code | pass@1 | v3 | 40 | 90.0% | [76.9%, 96.0%] | +0.0 | [-7.5, +7.5] | 1 | 1/1 | match | inconclusive |
| hardmath | strict | v21 | 100 | 76.0% | [66.8%, 83.3%] | ref |  |  |  |  |  |
| hardmath | strict | v3 | 100 | 89.0% | [81.4%, 93.7%] | +13.0 | [+6.0, +20.0] | 0.000977 | 1/14 | match | non-inferior, better |
| hardmath | number | v21 | 100 | 97.0% | [91.5%, 99.0%] | ref |  |  |  |  |  |
| hardmath | number | v3 | 100 | 97.0% | [91.5%, 99.0%] | +0.0 | [-3.0, +3.0] | 1 | 1/1 | match | inconclusive |

### pooled by family (each item once)

| family | metric | tag | n | score | 95% Wilson | Δ vs v21 (pts) | 95% paired bootstrap | McNemar p | ref-only/tag-only right | items | non-inferiority (lower ≥ −2 pts) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coding (humaneval+mbpp) | pass@1 | v21 | 421 | 97.9% | [96.0%, 98.9%] | ref |  |  |  |  |  |
| coding (humaneval+mbpp) | pass@1 | v3 | 421 | 97.1% | [95.1%, 98.4%] | -0.7 | [-2.1, +0.7] | 0.508 | 6/3 | 421 paired | inconclusive |
| bug-fixing (humanevalfix+quixbugs) | pass@1 | v21 | 195 | 95.9% | [92.1%, 97.9%] | ref |  |  |  |  |  |
| bug-fixing (humanevalfix+quixbugs) | pass@1 | v3 | 195 | 95.4% | [91.5%, 97.6%] | -0.5 | [-3.6, +2.6] | 1 | 6/5 | 195 paired | inconclusive |
| maths (gsm8k_full+gsm8k+math500) | strict | v21 | 1819 | 77.5% | [75.5%, 79.3%] | ref |  |  |  |  |  |
| maths (gsm8k_full+gsm8k+math500) | strict | v3 | 1819 | 89.7% | [88.2%, 91.0%] | +12.2 | [+10.6, +13.9] | 1.02e-50 | 18/240 | 1819 paired | non-inferior, better |
| maths (gsm8k_full+gsm8k+math500) | number | v21 | 1819 | 95.4% | [94.4%, 96.3%] | ref |  |  |  |  |  |
| maths (gsm8k_full+gsm8k+math500) | number | v3 | 1819 | 95.8% | [94.7%, 96.6%] | +0.3 | [-0.3, +1.0] | 0.418 | 16/22 | 1819 paired | non-inferior |
| knowledge (mmlu_2k+mmlu) | accuracy | v21 | 2338 | 87.5% | [86.1%, 88.7%] | ref |  |  |  |  |  |
| knowledge (mmlu_2k+mmlu) | accuracy | v3 | 2338 | 87.0% | [85.6%, 88.3%] | -0.4 | [-1.4, +0.6] | 0.437 | 72/62 | 2338 paired | non-inferior |

