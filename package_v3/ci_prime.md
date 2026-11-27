### per suite (repeat r0; 10,000 resamples, seed 1234)

| suite | metric | tag | n | score | 95% Wilson | Δ vs stock (pts) | 95% paired bootstrap | McNemar p | ref-only/tag-only right | items | non-inferiority (lower ≥ −2 pts) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pub_humaneval | pass@1 | stock | 164 | 99.4% | [96.6%, 99.9%] | ref |  |  |  |  |  |
| pub_humaneval | pass@1 | prime | 164 | 98.2% | [94.8%, 99.4%] | -1.2 | [-3.0, +0.0] | 0.5 | 2/0 | match | inconclusive |
| pub_mbpp | pass@1 | stock | 257 | 96.9% | [94.0%, 98.4%] | ref |  |  |  |  |  |
| pub_mbpp | pass@1 | prime | 257 | 96.5% | [93.5%, 98.1%] | -0.4 | [-2.3, +1.6] | 1 | 4/3 | match | inconclusive |
| pub_humanevalfix | pass@1 | stock | 164 | 96.3% | [92.2%, 98.3%] | ref |  |  |  |  |  |
| pub_humanevalfix | pass@1 | prime | 164 | 94.5% | [89.9%, 97.1%] | -1.8 | [-4.9, +0.6] | 0.375 | 4/1 | match | inconclusive |
| pub_quixbugs | pass@1 | stock | 31 | 96.8% | [83.8%, 99.4%] | ref |  |  |  |  |  |
| pub_quixbugs | pass@1 | prime | 31 | 96.8% | [83.8%, 99.4%] | +0.0 | [-9.7, +9.7] | 1 | 1/1 | match | inconclusive |
| pub_gsm8k_full | strict | stock | 1319 | 67.1% | [64.5%, 69.6%] | ref |  |  |  |  |  |
| pub_gsm8k_full | strict | prime | 1319 | 88.1% | [86.2%, 89.7%] | +21.0 | [+18.7, +23.4] | 8.86e-68 | 15/292 | match | non-inferior, better |
| pub_gsm8k_full | number | stock | 1319 | 96.3% | [95.1%, 97.2%] | ref |  |  |  |  |  |
| pub_gsm8k_full | number | prime | 1319 | 96.4% | [95.2%, 97.2%] | +0.1 | [-0.7, +0.8] | 1 | 13/14 | match | non-inferior |
| pub_gsm8k | strict | stock | 300 | 68.0% | [62.5%, 73.0%] | ref |  |  |  |  |  |
| pub_gsm8k | strict | prime | 300 | 86.3% | [82.0%, 89.8%] | +18.3 | [+13.7, +23.3] | 3.29e-14 | 3/58 | match | non-inferior, better |
| pub_gsm8k | number | stock | 300 | 96.7% | [94.0%, 98.2%] | ref |  |  |  |  |  |
| pub_gsm8k | number | prime | 300 | 98.0% | [95.7%, 99.1%] | +1.3 | [-0.3, +3.3] | 0.289 | 2/6 | match | non-inferior |
| pub_math500 | is_equiv | stock | 500 | 92.6% | [90.0%, 94.6%] | ref |  |  |  |  |  |
| pub_math500 | is_equiv | prime | 500 | 92.4% | [89.7%, 94.4%] | -0.2 | [-1.8, +1.4] | 1 | 8/7 | match | non-inferior |
| pub_mmlu_2k | accuracy | stock | 2000 | 89.6% | [88.2%, 90.9%] | ref |  |  |  |  |  |
| pub_mmlu_2k | accuracy | prime | 2000 | 89.5% | [88.1%, 90.8%] | -0.1 | [-1.1, +1.0] | 0.928 | 63/61 | match | non-inferior |
| pub_mmlu | accuracy | stock | 400 | 87.2% | [83.6%, 90.2%] | ref |  |  |  |  |  |
| pub_mmlu | accuracy | prime | 400 | 86.8% | [83.1%, 89.7%] | -0.5 | [-2.8, +1.8] | 0.832 | 12/10 | match | inconclusive |
| code | pass@1 | stock | 40 | 95.0% | [83.5%, 98.6%] | ref |  |  |  |  |  |
| code | pass@1 | prime | 40 | 92.5% | [80.1%, 97.4%] | -2.5 | [-10.0, +5.0] | 1 | 2/1 | match | inconclusive |
| hardmath | strict | stock | 100 | 73.0% | [63.6%, 80.7%] | ref |  |  |  |  |  |
| hardmath | strict | prime | 100 | 91.0% | [83.8%, 95.2%] | +18.0 | [+10.0, +26.0] | 4.01e-05 | 1/19 | match | non-inferior, better |
| hardmath | number | stock | 100 | 94.0% | [87.5%, 97.2%] | ref |  |  |  |  |  |
| hardmath | number | prime | 100 | 97.0% | [91.5%, 99.0%] | +3.0 | [-1.0, +8.0] | 0.375 | 1/4 | match | non-inferior |

### pooled by family (each item once)

| family | metric | tag | n | score | 95% Wilson | Δ vs stock (pts) | 95% paired bootstrap | McNemar p | ref-only/tag-only right | items | non-inferiority (lower ≥ −2 pts) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coding (humaneval+mbpp) | pass@1 | stock | 421 | 97.9% | [96.0%, 98.9%] | ref |  |  |  |  |  |
| coding (humaneval+mbpp) | pass@1 | prime | 421 | 97.1% | [95.1%, 98.4%] | -0.7 | [-2.1, +0.7] | 0.508 | 6/3 | 421 paired | inconclusive |
| bug-fixing (humanevalfix+quixbugs) | pass@1 | stock | 195 | 96.4% | [92.8%, 98.3%] | ref |  |  |  |  |  |
| bug-fixing (humanevalfix+quixbugs) | pass@1 | prime | 195 | 94.9% | [90.8%, 97.2%] | -1.5 | [-4.1, +1.0] | 0.453 | 5/2 | 195 paired | inconclusive |
| maths (gsm8k_full+gsm8k+math500) | strict | stock | 1819 | 74.1% | [72.0%, 76.1%] | ref |  |  |  |  |  |
| maths (gsm8k_full+gsm8k+math500) | strict | prime | 1819 | 89.3% | [87.8%, 90.6%] | +15.2 | [+13.4, +17.0] | 2.1e-62 | 23/299 | 1819 paired | non-inferior, better |
| maths (gsm8k_full+gsm8k+math500) | number | stock | 1819 | 95.3% | [94.2%, 96.2%] | ref |  |  |  |  |  |
| maths (gsm8k_full+gsm8k+math500) | number | prime | 1819 | 95.3% | [94.2%, 96.2%] | +0.0 | [-0.7, +0.7] | 1 | 21/21 | 1819 paired | non-inferior |
| knowledge (mmlu_2k+mmlu) | accuracy | stock | 2338 | 89.3% | [88.0%, 90.5%] | ref |  |  |  |  |  |
| knowledge (mmlu_2k+mmlu) | accuracy | prime | 2338 | 89.1% | [87.8%, 90.3%] | -0.2 | [-1.2, +0.8] | 0.74 | 75/70 | 2338 paired | non-inferior |

