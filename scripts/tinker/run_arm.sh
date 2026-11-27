#!/bin/bash
# Train one RL arm, then hand each logged sampler checkpoint to eval_ckpt.sh in the
# background and exit, so the training slot frees as soon as training ends.
#   scripts/tinker/run_arm.sh <tag> [train_rl.py flags...]
# Why a script: the harness refuses compound shell commands, and train->eval must chain.
set -u
TAG="$1"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
export LAB_JUDGE_MAX_USD="${LAB_JUDGE_MAX_USD:-3}"
cd "$HERE" || exit 1
"$PY" train_rl.py --tag "$TAG" "$@" > "$LAB/logs/train_$TAG.stdout" 2>&1
echo "train exit $?" >> "$LAB/logs/train_$TAG.stdout"
"$PY" - "$LAB/logs/train_$TAG.log" > "$LAB/logs/ckpts_$TAG.txt" <<'EOF'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("event") == "checkpoint" and r.get("sampler_path"):
        print(r["step"], r["sampler_path"])
EOF
while read -r STEP SPATH; do
  nohup "$HERE/eval_ckpt.sh" "$TAG" "$STEP" "$SPATH" > /dev/null 2>&1 &
done < "$LAB/logs/ckpts_$TAG.txt"
echo "arm $TAG trained $(date '+%H:%M'); evals launched" >> "$LAB/logs/train_$TAG.stdout"
