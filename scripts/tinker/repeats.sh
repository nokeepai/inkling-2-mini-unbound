#!/bin/bash
# Noise control: 3 independent t0.6 repeats of hard maths and the code eval.
#   scripts/tinker/repeats.sh <name> <sampler_path|base> <effort>
NAME="$1"; SPATH="$2"; EFF="$3"
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
MP=()
[ "$SPATH" != "base" ] && MP=(--model-path "$SPATH")
for R in 1 2 3; do
  "$PY" eval.py --tag "rep_${NAME}_e${EFF}_hardmath_r$R" --prompts prompts_hardmath.jsonl \
      --effort "$EFF" --temperature 0.6 "${MP[@]}" > "$LAB/logs/rep_${NAME}_hm_r$R.stdout" 2>&1 &
  "$PY" code_eval.py --tag "rep_${NAME}_e${EFF}_code_r$R" --effort "$EFF" "${MP[@]}" \
      > "$LAB/logs/rep_${NAME}_code_r$R.stdout" 2>&1 &
done
wait
echo "repeats $NAME e$EFF done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
