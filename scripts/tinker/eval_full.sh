#!/bin/bash
# Full sweep for a release candidate. Does not overwrite the light-eval files.
#   scripts/tinker/eval_full.sh <name> <sampler_path> [paid]
# Covers: main set at e0.9 and e0.3 (Bison + Haiku), hard agent eval (forced at both efforts,
# no-force at e0.3; no-force e0.9 comes from the light eval), long-system-prompt rules, easy
# agent eval, multi-turn roleplay, temperature-1.0 settings at e0.9, and 3 repeats of hard
# maths and code at e0.9. "paid" bills and records the sampling under the ledger cap.
NAME="$1"; SPATH="$2"
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
export LAB_JUDGE_MAX_USD="${LAB_JUDGE_MAX_USD:-6}" LAB_ALT_JUDGE_MAX_USD="${LAB_ALT_JUDGE_MAX_USD:-2}"
if [ "${3:-}" = paid ]; then
  : "${LAB_PAID_MAX_USD:?set LAB_PAID_MAX_USD}"
  export LAB_PAID_MAX_USD LAB_PAID_RUN="${NAME}_fulleval"
fi
for E in 0.9 0.3; do
  T="$NAME"; [ "$E" = 0.3 ] && T="${NAME}_e0.3"
  (
    "$PY" eval.py --tag "$T" --effort "$E" --temperature 0.6 --judge --model-path "$SPATH" \
        > "$LAB/logs/eval_$T.stdout" 2>&1
    "$PY" eval.py --tag "$T" --rejudge-alt > "$LAB/logs/rejudge_alt_$T.stdout" 2>&1
  ) &
  "$PY" agent_eval_hard.py --tag "${NAME}_agenthard_e${E}" --effort "$E" --model-path "$SPATH" \
      > "$LAB/logs/eval_${NAME}_agenthard_e${E}.stdout" 2>&1 &
  "$PY" sysrules_eval.py run --tag "${NAME}_sysrules_e${E}" --effort "$E" --model-path "$SPATH" \
      > "$LAB/logs/eval_${NAME}_sysrules_e${E}.stdout" 2>&1 &
done
"$PY" agent_eval_hard.py --tag "${NAME}_agenthard_noforce_e0.3" --effort 0.3 --no-force \
    --model-path "$SPATH" > "$LAB/logs/eval_${NAME}_agenthard_noforce_e0.3.stdout" 2>&1 &
"$HERE/agent_runs.sh" "$NAME" "$SPATH" &
"$HERE/rp_eval.sh" "${NAME}_rp" "$SPATH" &
"$HERE/repeats.sh" "$NAME" "$SPATH" 0.9 &
"$HERE/unsloth.sh" "$NAME" "$SPATH" 0.9 &
wait
echo "eval_full $NAME done $(date '+%H:%M')" >> "$LAB/logs/eval_ckpt.log"
