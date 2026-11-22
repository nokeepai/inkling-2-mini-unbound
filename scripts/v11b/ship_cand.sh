#!/bin/bash
# Mac: merge a LoRA on the res pod, move tensors to the eval pod via the PRIVATE HF scratch repo, build Q8_0 there,
# start its val screen.   bash ship_cand.sh <cand> <run_dir_on_res> <lora file> <merge_scale> <gpu>
set -u
C=$1; RUN=$2; LF=$3; MS=$4; GPU=$5
L="bash $HOME/Code/inkling-abl/scripts/ink_lium.sh"; RES=${RES_POD:-ink-v11-res2}; EV=${EVAL_POD:-ink-v11-eval2}
TOKV="$(cat $HOME/.cache/huggingface/token)"
$L exec $RES "cd /root/v11b; mkdir -p cands/$C; python3 merge_lora.py $RUN/$LF /root/w/v10 hf_name_map.json cands/$C/tensors.safetensors $MS 2>&1 | tail -1" 2>&1 | tail -1
$L exec $RES -e "HF_TOKEN=$TOKV" "hf upload $SCRATCH_REPO /root/v11b/cands/$C/tensors.safetensors cands/$C/tensors.safetensors --repo-type dataset --private > /dev/null 2>&1; echo UP_\$?" 2>&1 | tail -1
$L exec $EV -e "HF_TOKEN=$TOKV" "cd /root/v11b; nohup bash fetch_build_screen.sh $C $GPU > /root/logs/fbs_$C.log 2>&1 & echo STARTED_$C" 2>&1 | tail -1
