#!/bin/bash
# Multi-turn roleplay eval wrapper: scripts/tinker/rp_eval.sh <tag> [sampler_path]
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
export LAB_JUDGE_MAX_USD="${LAB_JUDGE_MAX_USD:-3}"
MP=()
[ -n "${2:-}" ] && MP=(--model-path "$2")
"$PY" "$HERE/rp.py" eval --tag "$1" "${MP[@]}" > "$LAB/logs/eval_$1.stdout" 2>&1
echo "rp_eval $1 exit $? $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
