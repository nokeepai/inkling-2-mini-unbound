#!/bin/bash
# ON POD: find which config gives finite perplexity on the base BF16 (2 chunks each).
export CUDA_VISIBLE_DEVICES=6,7
P=/data/llama.cpp/build-m/bin/llama-perplexity
M=${1:-/data/base/Inkling-2-Mini-BF16.gguf}
A="-m $M -f /data/work/wiki.test.raw -c 512 -b 512 --chunks 2"
run() { tag=$1; shift; $P $A "$@" > /data/logs/probe_$tag.log 2>&1; echo "$tag: $(grep -o -E '\[1\][^ ]*' /data/logs/probe_$tag.log | head -1) $(grep -E 'Final' /data/logs/probe_$tag.log)"; }
run fa_off -fa off
run cpu_only -ngl 0 -fa off
run q8ref_fit -fa off -m /data/base/Inkling-2-Mini-Q8_0.gguf
echo PROBE_DONE
