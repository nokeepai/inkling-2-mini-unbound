#!/bin/bash
# When <tag> logs its checkpoint for <step>, run main3.sh on it (three graded main-set passes,
# for the paired Q&A / creative / roleplay comparison). Free-tier checkpoints only.
#   scripts/tinker/queue_main3.sh <tag> <step>
TAG="$1"; STEP="$2"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
until grep -q "\"event\": \"checkpoint\", \"step\": $STEP," "$LAB/logs/train_$TAG.log" 2>/dev/null; do sleep 15; done
SPATH=$(grep "\"event\": \"checkpoint\", \"step\": $STEP," "$LAB/logs/train_$TAG.log" | "$LAB/.venv/bin/python" -c \
  'import json,sys; print(json.loads(sys.stdin.readline())["sampler_path"])')
"$HERE/main3.sh" "${TAG}_step${STEP}" "$SPATH"
