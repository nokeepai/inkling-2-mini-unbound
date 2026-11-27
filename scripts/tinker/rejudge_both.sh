#!/bin/bash
# Grade existing main-set evals with Bison then Haiku: scripts/tinker/rejudge_both.sh <tag>...
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
export LAB_JUDGE_MAX_USD="${LAB_JUDGE_MAX_USD:-3}"
for TAG in "$@"; do
  "$PY" eval.py --tag "$TAG" --rejudge > "$LAB/logs/rejudge_$TAG.stdout" 2>&1
  "$PY" eval.py --tag "$TAG" --rejudge-alt > "$LAB/logs/rejudge_alt_$TAG.stdout" 2>&1
  echo "rejudged $TAG $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
done
