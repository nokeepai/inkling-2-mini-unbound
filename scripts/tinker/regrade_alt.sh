#!/bin/bash
# Re-run the held-out (Haiku) grader on existing outputs whose grading stopped at the spend cap.
#   LAB_ALT_JUDGE_MAX_USD=4 scripts/tinker/regrade_alt.sh <tag> [<tag> ...]
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
cd "$(dirname "$0")" || exit 1
for T in "$@"; do
  "$PY" eval.py --tag "$T" --rejudge-alt > "$LAB/logs/rejudge_alt_$T.stdout" 2>&1 &
done
wait
echo "regrade_alt done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
