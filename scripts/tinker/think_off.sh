#!/bin/bash
# Thinking-off evals (effort 0 + forced answer start) at t0.6 and greedy: scripts/tinker/think_off.sh <name> [sampler_path]
NAME="$1"
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
MP=()
[ -n "${2:-}" ] && MP=(--model-path "$2")
"$PY" eval.py --tag "${NAME}_off_t06" --thinking off --temperature 0.6 "${MP[@]}" \
    > "$LAB/logs/eval_${NAME}_off_t06.stdout" 2>&1 &
"$PY" eval.py --tag "${NAME}_off_greedy" --thinking off --temperature 0 "${MP[@]}" \
    > "$LAB/logs/eval_${NAME}_off_greedy.stdout" 2>&1 &
wait
echo "think_off $NAME done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
