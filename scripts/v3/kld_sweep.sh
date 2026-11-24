#!/bin/bash
# ON eval pod: merge a LoRA stack at several scales of the LAST LoRA, build Q8_0 GGUFs, KLD each on one GPU.
#   bash kld_sweep.sh <name> <gpu> "<fixed stack prefix or ''>" <last_lora.pt> <scale1> [scale2 ...]
# -> /data/v11/<name>_s<scale>-Q8_0.gguf and /root/v3/ev/<name>_s<scale>/kld.txt ; summary in /root/v3/ev/sweep_<name>.txt
set -u
N=$1; G=$2; PRE=$3; LAST=$4; shift 4
cd /root/v11b
for s in "$@"; do
  C=${N}_s$s; mkdir -p /data/cands/$C
  python3 merge_stack.py /data/v2hf hf_name_map.json /data/cands/$C/tensors.safetensors $PRE $LAST:$s > /data/logs/merge_$C.log 2>&1 \
    || { echo "$C merge failed" >> /root/v3/ev/sweep_$N.txt; continue; }
  bash build_gguf.sh $C /data/cands/$C/tensors.safetensors > /data/logs/gguf_$C.log 2>&1
  rm -f /data/cands/$C/tensors.safetensors
  bash kld.sh $C /data/v11/$C-Q8_0.gguf $G > /dev/null 2>&1
  echo "$C $(grep -a 'Mean    KLD' /root/v3/ev/$C/kld.txt) $(grep -a 'Same top' /root/v3/ev/$C/kld.txt)" >> /root/v3/ev/sweep_$N.txt
done
echo SWEEP_DONE >> /root/v3/ev/sweep_$N.txt
