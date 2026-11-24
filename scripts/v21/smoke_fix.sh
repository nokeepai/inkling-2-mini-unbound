#!/bin/bash
pip install -q tiktoken sentencepiece > /dev/null 2>&1
cp -n /data/v2hf/tokenizer.json /data/v2hf/tokenizer_config.json /data/compare/models/tok/ 2>/dev/null
mkdir -p /data/compare/models/tok/tiktoken && cp -n /data/v2hf/tiktoken/tokenizer.model /data/compare/models/tok/tiktoken/ 2>/dev/null
bash /root/v21/smoke.sh /data/rel/gguf > /data/rel/logs/smoke.log 2>&1
grep SMOKE /data/rel/logs/smoke.log >> /root/logs/rel21.log
