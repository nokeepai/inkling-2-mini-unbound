#!/bin/bash
# Temperature-1.0 main set x3 with both graders (tags t1_<name>_r1..3). One pass has only 40
# creative prompts and swung 67% -> 82% for the same model, so single passes cannot rank models.
#   scripts/tinker/t1_reps.sh <name> <sampler_path|base> [paid]
NAME="$1"; SPATH="$2"
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
cd "$(dirname "$0")" || exit 1
export LAB_JUDGE_MAX_USD="${LAB_JUDGE_MAX_USD:-6}" LAB_ALT_JUDGE_MAX_USD="${LAB_ALT_JUDGE_MAX_USD:-4}"
if [ "${3:-}" = paid ]; then
  : "${LAB_PAID_MAX_USD:?set LAB_PAID_MAX_USD}"
  export LAB_PAID_MAX_USD LAB_PAID_RUN="${NAME}_t1"
fi
MP=(); [ "$SPATH" != "base" ] && MP=(--model-path "$SPATH")
for R in 1 2 3; do
  T="t1_${NAME}_r$R"
  (
    "$PY" eval.py --tag "$T" --effort 0.9 --temperature 1.0 --judge ${MP[@]+"${MP[@]}"} > "$LAB/logs/eval_$T.stdout" 2>&1
    "$PY" eval.py --tag "$T" --rejudge-alt > "$LAB/logs/rejudge_alt_$T.stdout" 2>&1
  ) &
done
wait
echo "t1_reps $NAME done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
