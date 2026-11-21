#!/bin/bash
# ON POD: verify the playground answers one chat call.
curl -s http://127.0.0.1:8080/health; echo
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":300,"temperature":0}' | head -c 1200; echo
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
