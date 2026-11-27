#!/bin/bash
# Start w2_div when w2_long's training slot frees (w3_code takes w2_attn's slot first).
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
until grep -q "train exit" "$LAB/logs/train_w2_long.stdout" 2>/dev/null; do sleep 30; done
echo "$(date '+%H:%M') w2_div starting (w2_long slot freed)" >> "$LAB/status.txt"
"$HERE/run_arm.sh" w2_div --div --lr 5e-5 --steps 20 --group 8 --prompts-per-step 16 --rank 32 \
    --loss importance_sampling --effort 0.9 --judge --save-every 20 --prompts prompts_train_hard.jsonl \
    --no-loop --no-overlong --no-length --no-brevity
