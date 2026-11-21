# Sources (pinned)

Downloaded 2026-11-21 by the data agent (scripts/data/fetch.py). Raw copies: sealed/raw/ (contain test prompts, so sealed).

| key | used as | url | pinned revision | licence | raw rows |
|---|---|---|---|---|---|
| mlabonne_harmful | harmful (fully exact-duplicated by AdvBench; contributes 0 rows) | https://huggingface.co/datasets/mlabonne/harmful_behaviors | `01cead01398926d81f7c52bdb790ee8cf77ebba7` | none declared on card (derived from AdvBench, MIT) | 520 |
| mlabonne_harmless | harmless | https://huggingface.co/datasets/mlabonne/harmless_alpaca | `02c6a92cfcf11bb0c387334f8146d149d65b587f` | none declared on card (derived from Alpaca, CC-BY-NC-4.0) | 31323 |
| alpaca | harmless (instruction + input when input non-empty) | https://huggingface.co/datasets/tatsu-lab/alpaca | `dce01c9b08f87459cf36a430d809084718273017` | cc-by-nc-4.0 | 52002 |
| orbench | borderline (or-bench-hard-1k.csv) | https://huggingface.co/datasets/bench-llm/or-bench | `e36d8b80e81837c8a8f264bbb2a49f1b32c7e272` | cc-by-4.0 | 1319 |
| sorrybench | harmful (question.jsonl, 440 base prompts; 80 in 8 excluded categories; 20 mutation files not used) | https://huggingface.co/datasets/sorry-bench/sorry-bench-202503 | `612a4e1f45db8adf884fa62318ddf9fa1c6e75e9` | other | 440 (360 after category exclusion) |
| mmlu | cap/mmlu300.jsonl (all/test) | https://huggingface.co/datasets/cais/mmlu | `c30699e8356da336a370243923dbaf21066bb9fe` | mit | 14042 |
| gsm8k | cap/gsm8k100.jsonl (main/test) | https://huggingface.co/datasets/openai/gsm8k | `740312add88f781978c0658806c59bc2815b9866` | mit | 1319 |
| advbench | harmful (goal column) | https://github.com/llm-attacks/llm-attacks | `098262edf85f807224e70ecd87b9d83716bf6b73` | MIT | 520 |
| harmbench | harmful (standard + contextual; copyright excluded) | https://github.com/centerforaisafety/HarmBench | `8e1604d1171fe8a48d8febecd22f600e462bdcdd` | MIT | 300 |
| strongreject | harmful | https://github.com/alexandrasouly/strongreject | `f7cad6c17e624e21d8df2278e918ae1dddb4cb56` | MIT | 313 |
| xstest | borderline (label == safe only) | https://github.com/paul-rottger/exaggerated-safety | `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d` | CC-BY-4.0 | 250 |
| probes | probes (40, written by the data agent) | sealed/probes_source.jsonl | n/a | project-internal | 40 |
| embedder | near-dup dedupe | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | apache-2.0 | n/a |

GitHub raw files were fetched from `raw.githubusercontent.com/<repo>/<sha>/<path>`; HF files via `hf_hub_download(revision=<sha>)`.

SORRY-Bench: the gate was accepted by the account owner on 2026-11-21; 202503 pinned. Only `question.jsonl` (prompt_style == base) is used; category names from the repo's `meta_info.py`.

