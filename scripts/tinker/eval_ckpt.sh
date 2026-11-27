#!/bin/bash
# Full eval of one sampler checkpoint: main set with Bison --judge, then Haiku --rejudge-alt,
# and the hard-maths set in parallel. Sampling only; does not use a training slot.
#   scripts/tinker/eval_ckpt.sh <run_tag> <step> <sampler_path> [effort]
TAG="$1"; STEP="$2"; SPATH="$3"; EFF="${4:-0.9}"
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
export LAB_JUDGE_MAX_USD="${LAB_JUDGE_MAX_USD:-3}"
NAME="${TAG}_step${STEP}"
"$PY" eval.py --tag "${NAME}_hardmath" --prompts prompts_hardmath.jsonl --effort "$EFF" \
    --temperature 0.6 --model-path "$SPATH" > "$LAB/logs/eval_${NAME}_hardmath.stdout" 2>&1 &
"$PY" code_eval.py --tag "${NAME}_code" --effort "$EFF" --model-path "$SPATH" \
    > "$LAB/logs/eval_${NAME}_code.stdout" 2>&1 &
"$PY" eval.py --tag "$NAME" --effort "$EFF" --temperature 0.6 --judge --model-path "$SPATH" \
    > "$LAB/logs/eval_${NAME}.stdout" 2>&1
"$PY" eval.py --tag "$NAME" --rejudge-alt > "$LAB/logs/rejudge_alt_${NAME}.stdout" 2>&1
# Hard agentic eval (standard since W2): forced-answer and no-force, at e0.9 and e0.3.
for E in 0.9 0.3; do
  "$PY" agent_eval_hard.py --tag "${NAME}_agenthard_e${E}" --effort "$E" --model-path "$SPATH" \
      > "$LAB/logs/eval_${NAME}_agenthard_e${E}.stdout" 2>&1 &
  "$PY" agent_eval_hard.py --tag "${NAME}_agenthard_noforce_e${E}" --effort "$E" --no-force \
      --model-path "$SPATH" > "$LAB/logs/eval_${NAME}_agenthard_noforce_e${E}.stdout" 2>&1 &
  "$PY" sysrules_eval.py run --tag "${NAME}_sysrules_e${E}" --effort "$E" --model-path "$SPATH" \
      > "$LAB/logs/eval_${NAME}_sysrules_e${E}.stdout" 2>&1 &
done
wait
echo "eval_ckpt $NAME done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
