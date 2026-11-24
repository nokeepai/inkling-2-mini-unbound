#!/bin/bash
# ON ink-v3-eval. HF_TOKEN from env (private scratch repo only). Fast path first: tok, wiki, llama.cpp, base+v2 Q8_0.
set -u
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p /data/base /data/orig /data/v10 /data/v2 /data/v3 /data/logs /data/hfs /data/compare/models/tok /data/compare/kld /root/compare/logs /root/compare/out /root/logs
echo "START $(date -u +%FT%TZ)"
pip install -q transformers safetensors numpy pyarrow hf_transfer sentencepiece "huggingface_hub[cli]" 2>&1 | tail -1
( hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da --include "tokenizer*" "*.jinja" "special_tokens_map.json" "config.json" "generation_config.json" "chat_template*" --local-dir /data/compare/models/tok > /dev/null 2>&1
  hf download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/test-*" --local-dir /data/compare/models/wikitext > /dev/null 2>&1
  cd /root/v11b && python3 prep_wiki.py; echo "DL_SMALL done $(date -u +%T)" ) &
( hf download nokeepai/Inkling-2-Mini-Unbound-GGUF Inkling-2-Mini-Unbound-v2-Q8_0.gguf --local-dir /data/v2 > /data/logs/dl_v2.log 2>&1; echo "DL_V2 rc=$? $(date -u +%T)" ) &
( hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-Q8_0.gguf --local-dir /data/base > /data/logs/dl_baseq8.log 2>&1; echo "DL_BASEQ8 rc=$? $(date -u +%T)" ) &
hf download flashvenom/v11b-scratch llama_19bc.tar.gz --repo-type dataset --local-dir /data/hfs > /dev/null 2>&1
hf download flashvenom/v11b-scratch --include 'orig_attn/*' --repo-type dataset --local-dir /data/hfs > /dev/null 2>&1
cp /data/hfs/orig_attn/* /data/orig/; ls /data/orig
(apt-get install -y -q cmake > /dev/null 2>&1 || pip install -q cmake)
bash /root/v11b/eval_build.sh /data/hfs/llama_19bc.tar.gz
wait
ls -la /data/base /data/v2 /data/compare/wiki.test.raw
echo "FAST_DONE $(date -u +%FT%TZ)"
