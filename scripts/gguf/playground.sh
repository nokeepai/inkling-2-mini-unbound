#!/bin/bash
# ON POD: long-lived owner playground, Unbound Q8_0 + vision projector, GPU 7, localhost:8080.
mkdir -p /root/logs
export CUDA_VISIBLE_DEVICES=7
exec /data/llama.cpp/build-m/bin/llama-server -m /data/release/Inkling-2-Mini-Unbound-Q8_0.gguf \
  --mmproj /data/base/mmproj-Inkling-2-Mini-Q8_0.gguf --jinja -c 32768 -ngl 99 --host 127.0.0.1 --port 8080
