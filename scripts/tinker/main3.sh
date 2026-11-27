#!/bin/bash
# Main set x3 at e0.9 with both graders (tags <name>, <name>_main_r2, <name>_main_r3), for the
# paired Q&A / creative / roleplay comparison against stock. Optional "paid" as in eval_full.sh.
#   scripts/tinker/main3.sh <name> <sampler_path> [paid]
NAME="$1"; SPATH="$2"
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
export LAB_JUDGE_MAX_USD="${LAB_JUDGE_MAX_USD:-6}" LAB_ALT_JUDGE_MAX_USD="${LAB_ALT_JUDGE_MAX_USD:-3}"
if [ "${3:-}" = paid ]; then
  : "${LAB_PAID_MAX_USD:?set LAB_PAID_MAX_USD}"
  export LAB_PAID_MAX_USD LAB_PAID_RUN="${NAME}_main3"
fi
for T in "$NAME" "${NAME}_main_r2" "${NAME}_main_r3"; do
  (
    "$PY" eval.py --tag "$T" --effort 0.9 --temperature 0.6 --judge --model-path "$SPATH" \
        > "$LAB/logs/eval_$T.stdout" 2>&1
    "$PY" eval.py --tag "$T" --rejudge-alt > "$LAB/logs/rejudge_alt_$T.stdout" 2>&1
  ) &
done
wait
echo "main3 $NAME done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
