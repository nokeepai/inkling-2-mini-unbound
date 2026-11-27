#!/bin/bash
# Agentic debugging eval (hidden tests) at e0.9 and e0.3: scripts/tinker/agent_runs.sh <name> [sampler_path]
NAME="$1"
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
MP=()
[ -n "${2:-}" ] && MP=(--model-path "$2")
for E in 0.9 0.3; do
  "$PY" agent_eval.py --tag "${NAME}_agent_e${E}" --effort "$E" "${MP[@]}" \
      > "$LAB/logs/eval_${NAME}_agent_e${E}.stdout" 2>&1 &
done
wait
echo "agent $NAME done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
