#!/bin/bash
# ON eval pod: candidate tensors -> patched BF16 GGUF (mapping proved byte-for-byte) -> Q8_0 (llama-quantize, as ggml-org).
#   bash build_gguf.sh <name> <tensors.safetensors>     -> /data/v11/<name>-Q8_0.gguf
set -u
N=$1; T=$2; B=/data/llama.cpp/build/bin; mkdir -p /data/v11
python3 /root/v11b/make_map_v11b.py $T /data/v11/$N.map.json || exit 1
python3 /root/v11b/patch_v11b.py /data/base/Inkling-2-Mini-BF16.gguf /data/orig $T /data/v11/$N.map.json /data/v11/$N-BF16.gguf "Inkling-2-Mini-Unbound-v1.1-$N" > /data/logs/patch_$N.log 2>&1 || { tail -3 /data/logs/patch_$N.log; exit 1; }
tail -1 /data/logs/patch_$N.log
$B/llama-quantize /data/v11/$N-BF16.gguf /data/v11/$N-Q8_0.gguf Q8_0 > /data/logs/quant_$N.log 2>&1 || { tail -3 /data/logs/quant_$N.log; exit 1; }
rm -f /data/v11/$N-BF16.gguf
ls -la /data/v11/$N-Q8_0.gguf && echo "GGUF_DONE $N"
