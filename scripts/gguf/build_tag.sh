#!/bin/bash
# ON POD: build llama.cpp at a given ref in its own worktree, for bisecting the BF16 NaN.  build_tag.sh <ref> <dir>
set -u
REF=$1; D=$2
cd /data/llama.cpp
git worktree add -f "$D" "$REF" > /dev/null 2>&1 || true
cd "$D"
git rev-parse HEAD
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DLLAMA_BUILD_HTML=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release > /data/logs/cmake_$(basename $D).log 2>&1
cmake --build build -j 120 --target llama-perplexity > /data/logs/build_$(basename $D).log 2>&1
echo "BUILT rc=$?"
export CUDA_VISIBLE_DEVICES=6,7
./build/bin/llama-perplexity -m /data/base/Inkling-2-Mini-BF16.gguf -f /data/work/wiki.test.raw -c 512 -b 512 --chunks 2 > /data/logs/probe_$(basename $D).log 2>&1
echo "$REF: $(grep -o -E '\[1\][^ ]*' /data/logs/probe_$(basename $D).log | head -1)"
