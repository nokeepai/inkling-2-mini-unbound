#!/bin/bash
# Every 10 min: status line for the arms listed in active_arms.txt (appended to status.txt),
# plus any finished evals. Prints immediately when a training run ends. Exits on w2_done.
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
seen_done=""
while [ ! -f "$LAB/w2_done" ]; do
  "$PY" status.py $(cat "$LAB/active_arms.txt") 2>&1 | tail -1
  tail -3 "$LAB/logs/eval_ckpt.log" 2>/dev/null | tr '\n' ';'; echo
  for i in $(seq 1 20); do
    sleep 30
    d=$(grep -l "trained" "$LAB"/logs/train_w2_*.stdout 2>/dev/null | tr '\n' ' ')
    if [ "$d" != "$seen_done" ]; then seen_done="$d"; echo "TRAINED: $d"; break; fi
  done
done
