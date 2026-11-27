#!/bin/bash
# Hard agentic eval only (forced + no-force, e0.9 + e0.3) for an already-evaluated checkpoint.
#   scripts/tinker/agenthard_ckpt.sh <name> <sampler_path>
NAME="$1"; SPATH="$2"
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
for E in 0.9 0.3; do
  "$PY" agent_eval_hard.py --tag "${NAME}_agenthard_e${E}" --effort "$E" --model-path "$SPATH" \
      > "$LAB/logs/eval_${NAME}_agenthard_e${E}.stdout" 2>&1 &
  "$PY" agent_eval_hard.py --tag "${NAME}_agenthard_noforce_e${E}" --effort "$E" --no-force \
      --model-path "$SPATH" > "$LAB/logs/eval_${NAME}_agenthard_noforce_e${E}.stdout" 2>&1 &
done
wait
echo "agenthard_ckpt $NAME done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
