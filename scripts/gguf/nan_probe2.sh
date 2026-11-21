#!/bin/bash
# ON POD: more BF16 NaN probes (1 chunk each) on master build-m.
export CUDA_VISIBLE_DEVICES=6,7
P=/data/llama.cpp/build-m/bin/llama-perplexity
M=/data/base/Inkling-2-Mini-BF16.gguf
A="-f /data/work/wiki.test.raw -c 512 --chunks 1"
run() { tag=$1; shift; $P $A "$@" > /data/logs/probe2_$tag.log 2>&1; echo "$tag: $(grep -o -E '\[1\][^ ]*' /data/logs/probe2_$tag.log | head -1)"; }
run ub16 -m $M -b 512 -ub 16
run ub1 -m $M -b 512 -ub 1 --chunks 1 -c 128
run unbound_q8 -m /data/release/Inkling-2-Mini-Unbound-Q8_0.gguf -b 512
run unbound_bf16 -m /data/release/Inkling-2-Mini-Unbound-BF16.gguf -b 512
echo PROBE2_DONE
