#!/bin/bash
# Runs ON the pod: downloads comparison models (ggml-org BF16/Q8_0 come from gguf's /data/base) + wikitext + tokenizer.
set -x
pip install -q transformers safetensors numpy 2>&1 | tail -2
M=/data/compare/models; mkdir -p $M /root/compare/logs
hf download AtomicChat/Inkling-2-Mini-Uncensored-GGUF Inkling-2-Mini-Uncensored-LoRA.gguf README.md --local-dir $M/atomic
hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da --include "tokenizer*" "*.jinja" "special_tokens_map.json" "config.json" "generation_config.json" "chat_template*" --local-dir $M/tok
hf download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/test-*" --local-dir $M/wikitext
hf download AtomicChat/Inkling-2-Mini-Uncensored-GGUF Inkling-2-Mini-Uncensored-AD-Q4_K.gguf --local-dir $M/atomic
hf download IndrasMirror/Inkling-2-Mini-XQ-Uncensored Inkling-2-Mini-XQ-Uncensored.gguf README.md --local-dir $M/xq
hf download JoaoZaokk/Inkling-2-Mini-Heretic-GGUF Inkling-2-Mini-Heretic-UD-Q4_K_XL.gguf README.md --local-dir $M/heretic
echo DL_DONE
