#!/bin/bash
# ON eval pod: fetch candidate tensors from the private scratch repo, build Q8_0, start val screen. HF_TOKEN from env.
#   bash fetch_build_screen.sh <cand> <gpu>
set -u
C=$1; G=$2; mkdir -p /data/cands/$C
[ -f /data/cands/$C/tensors.safetensors ] || { hf download $SCRATCH_REPO cands/$C/tensors.safetensors --repo-type dataset --local-dir /data/hfs >/dev/null 2>&1; mv /data/hfs/cands/$C/tensors.safetensors /data/cands/$C/; }
cd /root/v11b; bash build_gguf.sh $C /data/cands/$C/tensors.safetensors > /root/logs/build_$C.log 2>&1; tail -n1 /root/logs/build_$C.log
bash screen.sh val_$C $G /data/v11/$C-Q8_0.gguf > /root/logs/scr_$C.log 2>&1
