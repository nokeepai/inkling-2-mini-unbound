#!/bin/bash
# Start w3_code as soon as w2_attn's training slot frees (it decides Release 1, so it goes first).
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
until grep -q "train exit" "$LAB/logs/train_w2_attn.stdout" 2>/dev/null; do sleep 30; done
echo "$(date '+%H:%M') w3_code starting (w2_attn slot freed)" >> "$LAB/status.txt"
"$HERE/run_arm.sh" w3_code --lr 5e-5 --steps 20 --group 8 --prompts-per-step 16 --rank 32 \
    --loss importance_sampling --effort 0.9 --judge --save-every 20 --prompts prompts_train_code.jsonl \
    --no-loop --no-overlong --no-length --no-brevity
