#!/bin/bash
# Checks for what shorter thinking might cost, with enough samples to tell:
#   very hard maths (AIME, 16K thinking cap) at e0.9 and e0.3;
#   long-system-prompt rules x3 at e0.9 (120 tasks instead of 40);
#   main set x2 more at e0.9 with both graders (Q&A has only 20 prompts per pass).
#   scripts/tinker/depth_checks.sh <name> <sampler_path|base> [paid]
NAME="$1"; SPATH="$2"
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
export LAB_JUDGE_MAX_USD="${LAB_JUDGE_MAX_USD:-6}" LAB_ALT_JUDGE_MAX_USD="${LAB_ALT_JUDGE_MAX_USD:-3}"
if [ "${3:-}" = paid ]; then
  : "${LAB_PAID_MAX_USD:?set LAB_PAID_MAX_USD}"
  export LAB_PAID_MAX_USD LAB_PAID_RUN="${NAME}_depth"
fi
MP=()
[ "$SPATH" != "base" ] && MP=(--model-path "$SPATH")
for E in 0.9 0.3; do
  "$PY" eval.py --tag "${NAME}_veryhard_e${E}" --prompts prompts_veryhard.jsonl --effort "$E" \
      --temperature 0.6 --think-cap 16000 --answer-cap 2048 "${MP[@]}" \
      > "$LAB/logs/eval_${NAME}_veryhard_e${E}.stdout" 2>&1 &
done
for R in 1 2 3; do
  "$PY" sysrules_eval.py run --tag "${NAME}_sysrules_e0.9_r$R" --effort 0.9 "${MP[@]}" \
      > "$LAB/logs/eval_${NAME}_sysrules_e0.9_r$R.stdout" 2>&1 &
done
for R in 2 3; do
  T="${NAME}_main_r$R"
  (
    "$PY" eval.py --tag "$T" --effort 0.9 --temperature 0.6 --judge "${MP[@]}" > "$LAB/logs/eval_$T.stdout" 2>&1
    "$PY" eval.py --tag "$T" --rejudge-alt > "$LAB/logs/rejudge_alt_$T.stdout" 2>&1
  ) &
done
wait
echo "depth_checks $NAME done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
