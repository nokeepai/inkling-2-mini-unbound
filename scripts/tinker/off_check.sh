#!/bin/bash
# Thinking-off main set (t0.6, both graders) for one free-tier checkpoint: tag <tag>_step<step>_off.
# Compare against base_off_t06.
#   scripts/tinker/off_check.sh <tag> <step>
TAG="$1"; STEP="$2"
LAB="$HOME/.inkling-abl/tinker_lab"; PY="$LAB/.venv/bin/python"
cd "$(dirname "$0")" || exit 1
export LAB_JUDGE_MAX_USD="${LAB_JUDGE_MAX_USD:-6}" LAB_ALT_JUDGE_MAX_USD="${LAB_ALT_JUDGE_MAX_USD:-4}"
SPATH=$(grep "\"event\": \"checkpoint\", \"step\": $STEP," "$LAB/logs/train_$TAG.log" | "$PY" -c \
  'import json,sys; print(json.loads(sys.stdin.readline())["sampler_path"])')
T="${TAG}_step${STEP}_off"
"$PY" eval.py --tag "$T" --thinking off --temperature 0.6 --judge --model-path "$SPATH" > "$LAB/logs/eval_$T.stdout" 2>&1
"$PY" eval.py --tag "$T" --rejudge-alt > "$LAB/logs/rejudge_alt_$T.stdout" 2>&1
echo "off_check $T done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
