#!/bin/bash
# One-off: temperature-1.0 repeats for stock and the four step-20 models being compared.
#   LAB_PAID_MAX_USD=45 scripts/tinker/t1_batch.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
LAB="$HOME/.inkling-abl/tinker_lab"
DIV=$(grep '"step": 20' "$LAB/logs/train_w9_full_div.log" | "$LAB/.venv/bin/python" -c \
  'import json,sys; print(json.loads(sys.stdin.readline())["sampler_path"])')
"$HERE/t1_reps.sh" base base &
"$HERE/t1_reps.sh" w9_full_div_step20 "$DIV" &
"$HERE/t1_reps.sh" w4_mix_step20 "tinker://ca5a16ec-412e-57d4-bc58-5337fe028e2f:train:0/sampler_weights/w4_mix-step20" paid &
"$HERE/t1_reps.sh" w5_attn_lr1e4_step20 "tinker://a84f0f41-5ac6-5ff5-8091-83cfac10e128:train:0/sampler_weights/w5_attn_lr1e4-step20" paid &
"$HERE/t1_reps.sh" w4_mix_lr1e4_step20 "tinker://641f49b4-3969-5b30-84be-8e9ee890ee26:train:0/sampler_weights/w4_mix_lr1e4-step20" paid &
wait
echo "t1_batch done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
