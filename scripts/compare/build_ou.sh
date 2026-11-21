#!/bin/bash
# ON pod (CPU only): patch OU's 20 wo_ud tensors into the ggml-org BF16 GGUF with gguf's patch logic (proves each
# mapping byte-for-byte against the original checkpoint first), llama-quantize to Q8_0 exactly as gguf does for ours.
# Control: also requantize the UNPATCHED BF16 and compare every tensor with ggml-org's published Q8_0.
set -u
D=/data/compare/ou; L=/root/compare/logs
Q=/data/llama.cpp/build-m/bin/llama-quantize; [ -x $Q ] || Q=/data/llama.cpp/build/bin/llama-quantize
echo "quantize binary: $Q" > $L/ou_build.log
cd /root/compare/bin
python3 patch_gguf_ou.py /data/base/Inkling-2-Mini-BF16.gguf /data/orig $D/tensors.safetensors /root/gguf/name_map.json $D/ou-BF16.gguf >> $L/ou_build.log 2>&1
echo "PATCH rc=$?" >> $L/ou_build.log
CUDA_VISIBLE_DEVICES="" $Q $D/ou-BF16.gguf $D/ou-Q8_0.gguf Q8_0 40 > $L/ou_quant.log 2>&1 &
CUDA_VISIBLE_DEVICES="" $Q /data/base/Inkling-2-Mini-BF16.gguf $D/orig-requant-Q8_0.gguf Q8_0 40 > $L/orig_requant.log 2>&1 &
wait
echo "QUANT_DONE" >> $L/ou_build.log
python3 cmp_tensors.py /data/base/Inkling-2-Mini-Q8_0.gguf $D/orig-requant-Q8_0.gguf >> $L/ou_build.log 2>&1
python3 cmp_tensors.py /data/base/Inkling-2-Mini-Q8_0.gguf $D/ou-Q8_0.gguf >> $L/ou_build.log 2>&1
rm -f $D/ou-BF16.gguf
echo "BUILD_DONE" >> $L/ou_build.log
