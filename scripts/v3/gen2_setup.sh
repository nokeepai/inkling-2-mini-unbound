#!/bin/bash
# ON ink-v3-gen2: sampling-only pod: llama.cpp, v2 Q8_0, tokenizer. HF_TOKEN from env (scratch tarball only).
set -u
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p /data/v2 /data/logs /data/hfs /data/compare/models/tok /root/logs /root/v3/gens
pip install -q transformers safetensors numpy hf_transfer sentencepiece tiktoken protobuf "huggingface_hub[cli]" 2>&1 | tail -1
( hf download nokeepai/Inkling-2-Mini-Unbound-GGUF Inkling-2-Mini-Unbound-v2-Q8_0.gguf --local-dir /data/v2 > /data/logs/dl_v2.log 2>&1; echo "DL_V2 rc=$?" ) &
( hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da tokenizer.json tokenizer_config.json chat_template.jinja special_tokens_map.json config.json generation_config.json --local-dir /data/compare/models/tok > /dev/null 2>&1; echo DL_TOK ) &
hf download flashvenom/v11b-scratch llama_19bc.tar.gz --repo-type dataset --local-dir /data/hfs > /dev/null 2>&1
(apt-get install -y -q cmake > /dev/null 2>&1 || pip install -q cmake)
bash /root/v11b/eval_build.sh /data/hfs/llama_19bc.tar.gz
wait
python3 -c 'from transformers import AutoTokenizer; print("TOK", len(AutoTokenizer.from_pretrained("/data/compare/models/tok")))'
bash /root/v11b/servers.sh /data/v2/Inkling-2-Mini-Unbound-v2-Q8_0.gguf v2 0,1,2,3,4,5,6,7 8 65536
echo "GEN2_READY $(date -u +%FT%TZ)"
