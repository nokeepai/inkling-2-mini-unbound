#!/bin/bash
# ON POD: clean build of llama.cpp at the checked-out commit (19bc41dbb) into build-m. UI assets disabled
# (LLAMA_BUILD_HTML=OFF) to avoid the parallel UI-asset race seen in build2.log; one target at a time for the UI-less build.
set -u
cd /data/llama.cpp
git rev-parse HEAD
cmake -B build-m -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DLLAMA_BUILD_HTML=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release > /data/logs/cmake_m.log 2>&1
echo "CMAKE rc=$?"
cmake --build build-m -j 220 --target llama-quantize > /data/logs/build_m.log 2>&1
echo "QUANTIZE_BUILT rc=$?"
cmake --build build-m -j 220 --target llama-server llama-perplexity llama-cli >> /data/logs/build_m.log 2>&1
echo "BUILD_M rc=$?"
ls -la build-m/bin | grep -E 'llama-(server|perplexity|quantize|cli)$'
