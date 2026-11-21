#!/bin/bash
# ON POD, CPU only. Quantize the patched BF16 into 6 types in parallel. Every type below Q8_0 uses the ggml-org imatrix
# and keeps attn_output (our edited matrices) at q8_0.
set -u
IN=/data/release/Inkling-2-Mini-Unbound-BF16.gguf
Q=/data/llama.cpp/build-m/bin/llama-quantize
IM=/data/base/imatrix.gguf
mkdir -p /data/logs/quant
CUDA_VISIBLE_DEVICES="" $Q $IN /data/release/Inkling-2-Mini-Unbound-Q8_0.gguf Q8_0 40 > /data/logs/quant/Q8_0.log 2>&1 &
for T in Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M; do
  CUDA_VISIBLE_DEVICES="" $Q --imatrix $IM --tensor-type attn_output=q8_0 $IN /data/release/Inkling-2-Mini-Unbound-$T.gguf $T 40 > /data/logs/quant/$T.log 2>&1 &
done
wait
for T in Q8_0 Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M; do echo "$T: $(tail -n 2 /data/logs/quant/$T.log | tr '\n' ' ')"; done
echo QUANT_DONE
