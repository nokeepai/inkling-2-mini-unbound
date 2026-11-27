#!/bin/bash
# Re-grade every main eval with the eval-only Haiku grader (no resampling).
# w1c/w1d tags are waited for, since their evals may still be running.
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
for TAG in base_t06_think_e09 base_greedy_think_e09 smoke5_step5_t06_think_e09 \
           w1a_step20 w1a_step40 w1b_step20 w1b_step40 w1c_step20 w1c_step40 w1d_step20 w1d_step40; do
  until [ -f "$LAB/eval/$TAG.summary.json" ]; do sleep 60; done
  "$PY" eval.py --tag "$TAG" --rejudge-alt > "$LAB/logs/rejudge_alt_$TAG.stdout" 2>&1
  echo "alt done $TAG exit $? $(date '+%H:%M')"
done
