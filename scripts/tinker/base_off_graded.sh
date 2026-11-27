#!/bin/bash
# Stock model, thinking off, t0.6, both graders: the reference for off_check.sh (tag base_off_t06_g).
LAB="$HOME/.inkling-abl/tinker_lab"; PY="$LAB/.venv/bin/python"
cd "$(dirname "$0")" || exit 1
export LAB_JUDGE_MAX_USD="${LAB_JUDGE_MAX_USD:-6}" LAB_ALT_JUDGE_MAX_USD="${LAB_ALT_JUDGE_MAX_USD:-4}"
T=base_off_t06_g
"$PY" eval.py --tag "$T" --thinking off --temperature 0.6 --judge > "$LAB/logs/eval_$T.stdout" 2>&1
"$PY" eval.py --tag "$T" --rejudge-alt > "$LAB/logs/rejudge_alt_$T.stdout" 2>&1
echo "off_check $T done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
