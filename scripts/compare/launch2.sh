#!/bin/bash
# Mac-side: launch a borderline shard of a model on a free GPU.  launch2.sh <name> <gpu> <model.gguf> <lora|-> <k/n>
NAME=$1; GPU=$2; MODEL=$3; LORA=$4; SHARD=$5
PORT=$((8100 + GPU))
TAG="${NAME}__bl${SHARD%%/*}"
bash /Users/nikshepsvn/Code/inkling-abl/scripts/ink_lium.sh exec ink-gguf-1 "CMP_METRICS=borderline CMP_SKIP_KLD=1 CMP_SHARD=$SHARD nohup bash /root/compare/bin/run_model2.sh $TAG $GPU $PORT $MODEL $LORA > /root/compare/logs/$TAG.run.log 2>&1 &"
