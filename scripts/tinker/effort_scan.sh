#!/bin/bash
# Effort-scaling eval (sampling only): main set (no judge) + hard maths at each effort.
#   scripts/tinker/effort_scan.sh <name> <sampler_path|base> [efforts...]
NAME="$1"; SPATH="$2"; shift 2
EFFORTS=("${@:-0.3 0.6 0.9}")
[ $# -eq 0 ] && EFFORTS=(0.3 0.6 0.9)
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
MP=()
[ "$SPATH" != "base" ] && MP=(--model-path "$SPATH")
for E in "${EFFORTS[@]}"; do
  "$PY" eval.py --tag "effort_${NAME}_e${E}" --effort "$E" --temperature 0.6 "${MP[@]}" \
      > "$LAB/logs/eval_effort_${NAME}_e${E}.stdout" 2>&1 &
  "$PY" eval.py --tag "effort_${NAME}_e${E}_hardmath" --prompts prompts_hardmath.jsonl \
      --effort "$E" --temperature 0.6 "${MP[@]}" \
      > "$LAB/logs/eval_effort_${NAME}_e${E}_hardmath.stdout" 2>&1 &
done
wait
echo "effort_scan $NAME done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
