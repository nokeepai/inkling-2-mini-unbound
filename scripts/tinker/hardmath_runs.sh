#!/bin/bash
# Hard-maths eval (t0.6, e0.9) on base and each W1 arm's step-40 sampler, waiting for
# each arm's checkpoint list (written by run_arm.sh after training) before starting it.
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
COMMON=(--prompts prompts_hardmath.jsonl --effort 0.9 --temperature 0.6)
"$PY" eval.py --tag base_hardmath "${COMMON[@]}" > "$LAB/logs/eval_base_hardmath.stdout" 2>&1 &
for ARM in w1a w1b w1c w1d; do
  (
    until grep -q "^40 " "$LAB/logs/ckpts_$ARM.txt" 2>/dev/null; do sleep 60; done
    SPATH=$(grep "^40 " "$LAB/logs/ckpts_$ARM.txt" | awk '{print $2}')
    "$PY" eval.py --tag "${ARM}_step40_hardmath" "${COMMON[@]}" --model-path "$SPATH" \
        > "$LAB/logs/eval_${ARM}_step40_hardmath.stdout" 2>&1
  ) &
done
wait
echo "hardmath all done $(date '+%H:%M')"
