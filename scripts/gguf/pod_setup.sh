#!/bin/bash
# Runs ON ink-gguf-1. Downloads inputs and builds llama.cpp (CUDA, sm_120) under /data (overlay disk; /root is FUSE).
set -u
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /data/base /data/orig /data/release /data/work /data/logs
echo "START $(date -u +%FT%TZ)"
python3 -c 'import torch; print("torch", torch.__version__)' || pip install -q torch
pip install -q safetensors numpy sentencepiece pyyaml requests
( hf download ggml-org/Inkling-2-Mini-GGUF --local-dir /data/base > /data/logs/dl_base.log 2>&1; echo "DL_BASE_DONE rc=$?" >> /data/logs/dl_base.log ) &
( hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da --include 'model*.safetensors*' '*.json' --local-dir /data/orig > /data/logs/dl_orig.log 2>&1; echo "DL_ORIG_DONE rc=$?" >> /data/logs/dl_orig.log ) &
cd /data
git clone -q https://github.com/ggml-org/llama.cpp /data/llama.cpp
cd /data/llama.cpp
TAG=$(git tag -l 'b*' | sed 's/^b//' | grep -E '^[0-9]+$' | sort -n | tail -1)
echo "LATEST_TAG b$TAG"
git checkout -q "b$TAG"
git rev-parse HEAD > /data/release/llama_cpp_commit.txt
echo "b$TAG" >> /data/release/llama_cpp_commit.txt
git log -1 --format='%H %cI %s' >> /data/release/llama_cpp_commit.txt
git log --oneline | grep -i -E '32704|nvfp4.*loop|loop.*nvfp4' | head -5
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /data/logs/cmake.log 2>&1
cmake --build build -j 200 --target llama-cli llama-server llama-perplexity llama-quantize llama-completion > /data/logs/build.log 2>&1
echo "BUILD rc=$?"
pip install -q -e /data/llama.cpp/gguf-py
wait
echo "SETUP_DONE $(date -u +%FT%TZ)"
