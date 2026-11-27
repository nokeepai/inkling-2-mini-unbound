#!/bin/bash
# When an arm logs its checkpoint for <step>, run the candidate sweep on it:
# eval_full.sh (main set, agent evals, rules, roleplay, temperature 1.0, repeats) and
# depth_checks.sh (very hard maths, rules x3, main set x2 more).
#   scripts/tinker/queue_final.sh <tag> <step> <paid|free>
TAG="$1"; STEP="$2"; MODE="$3"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$LAB/.venv/bin/python"
until grep -q "\"event\": \"checkpoint\", \"step\": $STEP," "$LAB/logs/train_$TAG.log" 2>/dev/null; do
  grep -q "train exit" "$LAB/logs/train_$TAG.stdout" 2>/dev/null && { sleep 5; \
    grep -q "\"event\": \"checkpoint\", \"step\": $STEP," "$LAB/logs/train_$TAG.log" || exit 1; }
  sleep 15
done
SPATH=$(grep "\"event\": \"checkpoint\", \"step\": $STEP," "$LAB/logs/train_$TAG.log" | "$PY" -c \
  'import json,sys; print(json.loads(sys.stdin.readline())["sampler_path"])')
P=(); [ "$MODE" = paid ] && P=(paid)
"$HERE/eval_full.sh" "${TAG}_step${STEP}" "$SPATH" ${P[@]+"${P[@]}"} &
"$HERE/depth_checks.sh" "${TAG}_step${STEP}" "$SPATH" ${P[@]+"${P[@]}"} &
wait
