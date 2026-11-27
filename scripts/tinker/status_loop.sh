#!/bin/bash
# Print a W1 status line every 10 min (also appended to status.txt); exits when all 8 evals exist.
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
while true; do
  "$PY" status.py w1a w1b w1c w1d 2>&1 | tail -1
  n=$(ls "$HOME"/.inkling-abl/tinker_lab/eval/w1?_step{20,40}.summary.json 2>/dev/null | wc -l)
  if [ "$n" -ge 8 ]; then echo "ALL 8 W1 EVALS DONE"; exit 0; fi
  sleep 600
done
