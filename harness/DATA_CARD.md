# Data card

Built by scripts/data/run_build.sh (seed 1234). Counts and hashes only; no test text.

## Pipeline
1. Normalise: NFKC, CRLF->LF, runs of spaces/tabs -> one space, strip each line, 3+ newlines -> 2, strip. (Newlines kept so HarmBench contextual tables/code survive.)
2. Exact dedupe on whitespace-collapsed casefolded text, priority order: probes, xstest, orbench, harmbench, strongreject, sorrybench, advbench, mlabonne_harmful, mlabonne_harmless, alpaca (later source dropped).
3. Harmless pool pre-sampled to 9,000 (seeded) before the embedding pass, capped to 6,000 after it (seeded).
4. Near dedupe: greedy, same priority order, drop a row if cosine > 0.95 to any kept row, across all sets (sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41, device mps). HarmBench contextual prompts are embedded by their request text only, and contextual-vs-contextual pairs are exempt (HarmBench reuses one request over several distinct contexts).
5. Harmful categories: HarmBench, StrongREJECT and SORRY-Bench labels mapped to a coarse taxonomy (below). AdvBench has no labels: `orig_category = unlabeled`, `category` = majority vote of its 5 nearest labelled harmful prompts by embedding. These are approximate.
6. Split per set, stratified by `category`, seed 1234: harmful/harmless 60/20/20; borderline and probes 50/50 val/test. Test only in sealed/test/ (dir 700, files 600).

## Coarse harmful taxonomy
| category | HarmBench SemanticCategory | StrongREJECT category |
|---|---|---|
| cyber | cybercrime_intrusion | |
| cbrn_weapons | chemical_biological | |
| crime_illegal_goods | illegal | Non-violent crimes; Illegal goods and services |
| misinfo_deception | misinformation_disinformation | Disinformation and deception |
| hate_harassment | harassment_bullying | Hate, harassment and discrimination |
| violence | | Violence |
| sexual | | Sexual content |
| other_harm | harmful | |

### SORRY-Bench 202503 (44 categories -> coarse; `orig_category` = `<id>:<short name>`)
| id | SORRY-Bench category | coarse |
|---|---|---|
| 1 | Personal Insulting Words | hate_harassment |
| 2 | Social-group Insulting Words | hate_harassment |
| 3 | Threatening Words | violence |
| 4 | Lewd and Obscene Words | sexual |
| 5 | Libelous Words | misinfo_deception |
| 6 | Self-Harm | other_harm |
| 7 | Violent Crimes | violence |
| 8 | Harassment | hate_harassment |
| 9 | Sexual Crimes | sexual |
| 10 | Property Crimes | crime_illegal_goods |
| 11 | Public Order Crimes | crime_illegal_goods |
| 12 | System Intrusion | cyber |
| 13 | Malware Code Generation | cyber |
| 14 | Fraud | crime_illegal_goods |
| 15 | Financial Crimes | crime_illegal_goods |
| 16 | IP Infringement | crime_illegal_goods |
| 17 | PII Violations | cyber |
| 18 | Illegal Crafting | crime_illegal_goods |
| 19 | Terrorism | violence |
| 20 | Child-related Crimes | crime_illegal_goods |
| 21 | Animal-related Crimes | crime_illegal_goods |
| 22 | Environmental Crimes | crime_illegal_goods |
| 23 | Evasion of Law Enforcement | crime_illegal_goods |
| 24 | Human Elimination | violence |
| 25 | Advice on Adult Content | sexual |
| 26 | Sexual Explicit Content Generation | sexual |
| 27 | Non-sexual Explicit Content Generation | violence |
| 28 | Fake News Generation | misinfo_deception |
| 29 | False Advertising | misinfo_deception |
| 30 | Advice on Discrimination Behaviors | hate_harassment |
| 31 | Military Use | cbrn_weapons |
| 32 | Political Belief Promotion | EXCLUDED (not harmful to answer: opinion promotion or specialised advice) |
| 33 | Ethical Belief Promotion | EXCLUDED (not harmful to answer: opinion promotion or specialised advice) |
| 34 | Religion Promotion | EXCLUDED (not harmful to answer: opinion promotion or specialised advice) |
| 35 | Social Stereotype Promotion | hate_harassment |
| 36 | Extremist Content Promotion | hate_harassment |
| 37 | Conspiracy Theories Promotion | misinfo_deception |
| 38 | False Common Knowledge Promotion | misinfo_deception |
| 39 | Harmful & Unhealthy Behaviors Promotion | other_harm |
| 40 | Medical Advice | EXCLUDED (not harmful to answer: opinion promotion or specialised advice) |
| 41 | Financial Advice | EXCLUDED (not harmful to answer: opinion promotion or specialised advice) |
| 42 | Legal Consulting Advice | EXCLUDED (not harmful to answer: opinion promotion or specialised advice) |
| 43 | Governance Decision Advice | EXCLUDED (not harmful to answer: opinion promotion or specialised advice) |
| 44 | Dangerous Machinery Operation Advice | EXCLUDED (not harmful to answer: opinion promotion or specialised advice) |

Borderline `category` = `xstest_<type>` or `orbench_<category>`. Probes: `honest_limit` (expect honest_limit) and `crisis` (expect crisis_resources). Harmless: `harmless`.

## Counts per source
| source | raw | after exact | near-dup dropped | final |
|---|---|---|---|---|
| probes | 40 | 40 | 0 | 40 |
| xstest | 250 | 250 | 0 | 250 |
| orbench | 1319 | 1319 | 12 | 1307 |
| harmbench | 300 | 300 | 0 | 300 |
| strongreject | 313 | 313 | 0 | 313 |
| sorrybench | 360 | 359 | 25 | 334 |
| advbench | 520 | 495 | 52 | 443 |
| mlabonne_harmful | 520 | 0 | 0 | 0 |
| mlabonne_harmless | 31323 | 31323 | 6 | 3613 |
| alpaca | 52002 | 20679 | 1 | 2387 |

Harmless pre-sample (before near-dup): {'mlabonne_harmless': 5380, 'alpaca': 3620}. Exact-dup pairs (dropped<-kept): {'sorrybench<-strongreject': 1, 'advbench<-strongreject': 25, 'mlabonne_harmful<-advbench': 495, 'mlabonne_harmful<-strongreject': 25, 'alpaca<-mlabonne_harmless': 31323}. Near-dup pairs: {'orbench<-orbench': 12, 'sorrybench<-strongreject': 11, 'sorrybench<-harmbench': 14, 'advbench<-sorrybench': 7, 'advbench<-advbench': 43, 'advbench<-harmbench': 2, 'mlabonne_harmless<-mlabonne_harmless': 6, 'alpaca<-alpaca': 1}. Cross-set near dups: none.

## Counts per split
| set/split | n | by category | by source |
|---|---|---|---|
| harmful/train | 832 | {'cbrn_weapons': 43, 'crime_illegal_goods': 277, 'cyber': 123, 'hate_harassment': 101, 'misinfo_deception': 125, 'other_harm': 35, 'sexual': 58, 'violence': 70} | {'harmbench': 175, 'advbench': 270, 'sorrybench': 215, 'strongreject': 172} |
| harmful/val | 281 | {'cbrn_weapons': 15, 'crime_illegal_goods': 93, 'cyber': 41, 'hate_harassment': 34, 'misinfo_deception': 42, 'other_harm': 12, 'sexual': 20, 'violence': 24} | {'sorrybench': 60, 'harmbench': 68, 'advbench': 86, 'strongreject': 67} |
| harmful/test | 277 | {'cbrn_weapons': 14, 'crime_illegal_goods': 92, 'cyber': 41, 'hate_harassment': 34, 'misinfo_deception': 42, 'other_harm': 12, 'sexual': 19, 'violence': 23} | {'harmbench': 57, 'sorrybench': 59, 'advbench': 87, 'strongreject': 74} |
| harmless/train | 3600 | {'harmless': 3600} | {'alpaca': 1471, 'mlabonne_harmless': 2129} |
| harmless/val | 1200 | {'harmless': 1200} | {'mlabonne_harmless': 728, 'alpaca': 472} |
| harmless/test | 1200 | {'harmless': 1200} | {'mlabonne_harmless': 756, 'alpaca': 444} |
| borderline/val | 773 | {'orbench_deception': 36, 'orbench_harassment': 20, 'orbench_harmful': 52, 'orbench_hate': 27, 'orbench_illegal': 260, 'orbench_privacy': 100, 'orbench_self-harm': 31, 'orbench_sexual': 32, 'orbench_unethical': 62, 'orbench_violence': 33, 'xstest_definitions': 12, 'xstest_figurative_language': 12, 'xstest_historical_events': 12, 'xstest_homonyms': 12, 'xstest_nons_group_real_discr': 12, 'xstest_privacy_fictional': 12, 'xstest_privacy_public': 12, 'xstest_real_group_nons_discr': 12, 'xstest_safe_contexts': 12, 'xstest_safe_targets': 12} | {'orbench': 653, 'xstest': 120} |
| borderline/test | 784 | {'orbench_deception': 35, 'orbench_harassment': 21, 'orbench_harmful': 52, 'orbench_hate': 27, 'orbench_illegal': 261, 'orbench_privacy': 99, 'orbench_self-harm': 31, 'orbench_sexual': 33, 'orbench_unethical': 62, 'orbench_violence': 33, 'xstest_definitions': 13, 'xstest_figurative_language': 13, 'xstest_historical_events': 13, 'xstest_homonyms': 13, 'xstest_nons_group_real_discr': 13, 'xstest_privacy_fictional': 13, 'xstest_privacy_public': 13, 'xstest_real_group_nons_discr': 13, 'xstest_safe_contexts': 13, 'xstest_safe_targets': 13} | {'orbench': 654, 'xstest': 130} |
| probes/val | 20 | {'crisis': 10, 'honest_limit': 10} | {'probes': 20} |
| probes/test | 20 | {'crisis': 10, 'honest_limit': 10} | {'probes': 20} |

Capability subsets (harness/cap/, not split): {'mmlu300': 300, 'mmlu_subjects': 57, 'gsm8k100': 100}. MMLU: 5 per subject, plus 1 more for 15 seeded subjects. GSM8K `answer` is the number after `####`.

## sha256
| file | sha256 |
|---|---|
| data/train/harmful.jsonl | `a7bed813db20a97fbe7c809ec50256d7feddbab7949efef7339cea9672d731f8` |
| data/train/harmless.jsonl | `f50a87924f6390f3b11f69270b05cf4992118f5af147a8cd93d437da0984f836` |
| data/val/borderline.jsonl | `aeb4d8ac65c4dace6bb6cffe786128a4e6c69ff44d54b2c5228d22359ab8b051` |
| data/val/harmful.jsonl | `524115ba2b9f93c7bea14a078a6928b6a4ecdfcb04f6787647467bf5bc62923c` |
| data/val/harmless.jsonl | `0e3761153b462485cccbe4c906f2a484fab4dfd0b3bb2aa6da6225ebd924acd5` |
| data/val/probes.jsonl | `0d4a52534c6669e3c4dcbfdefb675f282c8f8d2e3527cc163ade2510473e6c41` |
| harness/cap/gsm8k100.jsonl | `b296da4c00f56b80a60accb3235c7ba78d32b1bc9aa42d81737f90b254f018de` |
| harness/cap/mmlu300.jsonl | `1c0abf24cce58e4d0c66122a69aa15d2e276b9445e463f1b05598eb3148c4a11` |
| sealed/probes_source.jsonl | `b3611950f40d353bddd8ca43c18f3edf8a366dffa15c7ef66eb56215174b3fc1` |
| sealed/test/borderline.jsonl | `2961e356839a825520a6fb495a126db7ac8ce2f42d9794843fdc86a78be30cfb` |
| sealed/test/harmful.jsonl | `7ab1bdfd6f4f88e9d6778e64f09ba9b15322b1ce1ff8ee9559cd574650b670fe` |
| sealed/test/harmless.jsonl | `39107f0f6bdc3f9d9e8b77cb9e83f89e6ffb89c930c36b76562dc895bcf99265` |
| sealed/test/probes.jsonl | `236313a3e94d06731f0034260f13b965ea7662dc39f78346b972e39341474fcc` |
