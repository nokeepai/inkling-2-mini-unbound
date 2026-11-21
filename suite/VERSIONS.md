# Phase 4 suite: pinned versions (agent `suite`, recorded 2026-11-21 15:40Z on ink-suite-1)

## Engine
- SGLang image `lmsysorg/sglang:inkling-2-mini`, image id `sha256:aff54906ae2e...`, amd64 manifest
  `sha256:421aed3ac11741381eac350bf7b1db28a0cafb6ae49b57ae78fba86a16a32306` (= ENGINE.md / state/pins.json).
- Flags: ENGINE.md frozen set (scripts/suite/suite_serve.sh). GPU RTX 5090 32 GB, driver 615.71.09, one server per GPU, --tp 1.
- Image has `--speculative-algorithm DSPARK`; the checkpoint carries `mtp.safetensors` + `mtp_config` (block_size 8).

## Host scorer environment (python 3.12.3)
| Package | Version |
|---|---|
| transformers | 5.24.0.dev0 = commit a10ebc3c1c44b01d0ee825d48b8d0de3253e4539 (GitHub archive zip; tokenizer + chat template only) |
| lm_eval | 0.4.9.1 (IFEval checkers: lm_eval.tasks.ifeval.utils.process_results) |
| evalplus | 0.3.1 (HumanEvalPlus-v0.1.10; `python -m evalplus.evaluate --dataset humaneval`) |
| math-verify | 0.8.0 (antlr4 4.13.2 extra) |
| nltk | 3.10.4 (punkt, punkt_tab) |
| langdetect / immutabledict | 1.0.9 / 4.3.1 |
| torch / safetensors / pandas | 2.12.0+cu130 / 0.8.0 / 3.1.1 |

## Data (normalised by scripts/suite/pod/prep_data.py; md5 of /srv/suite/data/*.jsonl)
| Bench | Source | n | md5 |
|---|---|---|---|
| MMLU-Pro | TIGER-Lab/MMLU-Pro test (HF main) | 12,032 | b152ca8cf80c52efaf14e56a52570774 |
| GSM8K | openai/gsm8k main test | 1,319 | 310ff05c244c97f4d63d07c227aabb38 |
| MATH-500 | HuggingFaceH4/MATH-500 test.jsonl | 500 | 6417a9a45f25047c98b3cb0bc18323e9 |
| GPQA Diamond | fingertap/GPQA-Diamond (public mirror, options pre-shuffled A-D; Idavidrein/gpqa is gated for our token) | 198 | 00fa6842d3f9e812e0df86f3fcf1a54f |
| IFEval | google/IFEval | 541 | 2e12bd2c753038be9292232291e5746c |
| HumanEval+ | evalplus HumanEvalPlus v0.1.10 | 164 | (evalplus cache) |
| RULER (deferred) | lm-eval 0.4.9.1 generators, seed 1234; see suite/VERSIONS_ruler.md | 325 + 130 | suite/ruler/data/SHA256SUMS |

Bootstrap: 10,000 resamples, seed 1234 (scripts/suite/pod/suite_common.py `boot`, same as harness/sgl_eval.py).
