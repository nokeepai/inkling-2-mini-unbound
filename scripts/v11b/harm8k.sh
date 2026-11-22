#!/bin/bash
# ON eval pod: harmful val/test (off + e0.9), answer cap 8192 (owner rule: effectively uncapped; think cap 4096 as the
# bench). ctx 49152 / np 6 so long answers fit in the unified KV pool. No KLD.
#   bash harm8k.sh <out_name> <gpu> <model.gguf> [data_dir=/root/v11b] [lora.gguf]
set -u
N=$1; G=$2; M=$3; D=${4:-/root/v11b}; LO=${5:--}
export CMP_DATA=$D CMP_TOK=/data/compare/models/tok CMP_METRICS=harmful CMP_ANS_CAP=8192 CMP_SKIP_KLD=1 CMP_CTX=49152 CMP_NP=6
cd /root/compare/bin && bash run_model.sh $N $G $((8200+G)) "$M" "$LO"
