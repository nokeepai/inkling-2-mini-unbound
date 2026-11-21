#!/bin/bash
# ON POD: find the largest ubatch where the BF16 GGUF gives finite perplexity (1 chunk each).
export CUDA_VISIBLE_DEVICES=6,7
P=/data/llama.cpp/build-m/bin/llama-perplexity
M=/data/base/Inkling-2-Mini-BF16.gguf
for ub in 32 64 128 256; do
  $P -f /data/work/wiki.test.raw -c 512 --chunks 1 -b 512 -ub $ub -m $M > /data/logs/probe3_$ub.log 2>&1
  echo "ub$ub: $(grep -o -E '\[1\][^ ]*' /data/logs/probe3_$ub.log | head -1)"
done
$P -f /data/work/wiki.test.raw -c 512 --chunks 1 -b 512 -ub 512 -ngl 0 -m /data/base/Inkling-2-Mini-Q8_0.gguf > /data/logs/probe3_q8cpu.log 2>&1
echo "q8 cpu ub512: $(grep -o -E '\[1\][^ ]*' /data/logs/probe3_q8cpu.log | head -1)"
echo PROBE3_DONE
