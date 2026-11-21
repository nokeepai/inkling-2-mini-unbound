#!/bin/bash
# Mac-side: launch run_model.sh on the pod in the background.  launch.sh <name> <gpu> <model.gguf> [lora|-] [--base-kld-first]
NAME=$1; GPU=$2; MODEL=$3; LORA=${4:--}; MODE=${5:-}
PORT=$((8100 + GPU))
bash /Users/nikshepsvn/Code/inkling-abl/scripts/ink_lium.sh exec ink-gguf-1 "nohup bash /root/compare/bin/run_model.sh $NAME $GPU $PORT $MODEL $LORA $MODE > /root/compare/logs/$NAME.run.log 2>&1 &"
