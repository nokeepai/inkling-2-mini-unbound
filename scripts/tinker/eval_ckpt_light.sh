#!/bin/bash
# Light checkpoint eval: the checks that decide the stall fix, fast and cheap.
#   no-force hard agent eval at e0.9 (the stall), code pass@1 at e0.9, hard maths at e0.9.
#   scripts/tinker/eval_ckpt_light.sh <run_tag> <step> <sampler_path>
# The full sweep (main set + judges, e0.3, rules) runs only on checkpoints worth it: eval_ckpt.sh.
TAG="$1"; STEP="$2"; SPATH="$3"
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
NAME="${TAG}_step${STEP}"
"$PY" agent_eval_hard.py --tag "${NAME}_agenthard_noforce_e0.9" --effort 0.9 --no-force \
    --model-path "$SPATH" > "$LAB/logs/eval_${NAME}_agenthard_noforce_e0.9.stdout" 2>&1 &
"$PY" code_eval.py --tag "${NAME}_code" --effort 0.9 --model-path "$SPATH" \
    > "$LAB/logs/eval_${NAME}_code.stdout" 2>&1 &
"$PY" eval.py --tag "${NAME}_hardmath" --prompts prompts_hardmath.jsonl --effort 0.9 \
    --temperature 0.6 --model-path "$SPATH" > "$LAB/logs/eval_${NAME}_hardmath.stdout" 2>&1 &
wait
echo "eval_light $NAME done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
