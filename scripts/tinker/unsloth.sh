#!/bin/bash
# Unsloth-recommended sampling (temp 1.0, top_p 1.0, top_k off = Tinker defaults) on the
# main set (Bison + Haiku graded), hard maths and code, at the given efforts.
#   scripts/tinker/unsloth.sh <name> <sampler_path|base> [efforts...]
NAME="$1"; SPATH="$2"; shift 2
EFFORTS=("$@"); [ ${#EFFORTS[@]} -eq 0 ] && EFFORTS=(0.9 0.3)
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
export LAB_JUDGE_MAX_USD="${LAB_JUDGE_MAX_USD:-3}"
MP=()
[ "$SPATH" != "base" ] && MP=(--model-path "$SPATH")
for E in "${EFFORTS[@]}"; do
  T="unsloth_${NAME}_e${E}"
  (
    "$PY" eval.py --tag "$T" --effort "$E" --temperature 1.0 --judge "${MP[@]}" > "$LAB/logs/eval_$T.stdout" 2>&1
    "$PY" eval.py --tag "$T" --rejudge-alt > "$LAB/logs/rejudge_alt_$T.stdout" 2>&1
  ) &
  "$PY" eval.py --tag "${T}_hardmath" --prompts prompts_hardmath.jsonl --effort "$E" --temperature 1.0 \
      "${MP[@]}" > "$LAB/logs/eval_${T}_hardmath.stdout" 2>&1 &
  "$PY" code_eval.py --tag "${T}_code" --effort "$E" --temperature 1.0 "${MP[@]}" \
      > "$LAB/logs/eval_${T}_code.stdout" 2>&1 &
done
wait
echo "unsloth $NAME done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
