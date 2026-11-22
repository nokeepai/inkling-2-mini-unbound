#!/bin/bash
# ON eval pod: build llama.cpp @ 19bc41dbb from a tarball (GitHub is slow from this pod). Flags as compare/gguf.
# The slow git clone (pid in $2, optional) is stopped by pid only.
set -u
T=$1; CLONE_PID=${2:-}
[ -n "$CLONE_PID" ] && kill "$CLONE_PID" 2>/dev/null
rm -rf /data/llama.cpp /data/llama_s; mkdir -p /data/llama.cpp
tar xzf "$T" -C /data/llama.cpp --strip-components=1
cd /data/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DLLAMA_BUILD_HTML=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release > /data/logs/cmake.log 2>&1
echo "CMAKE rc=$?"
cmake --build build -j 200 --target llama-quantize llama-server llama-perplexity > /data/logs/build.log 2>&1
echo "BUILD rc=$?"
ls -la build/bin | grep -E 'llama-(server|perplexity|quantize)$'
pip install -q -e /data/llama.cpp/gguf-py 2>&1 | tail -1
python3 -c 'import gguf; print("gguf-py", gguf.__file__)'
echo BUILD_DONE
