#!/bin/bash
# Train one arm and run the light checkpoint eval on every checkpoint as soon as it is saved
# (no waiting for training to end). Mode "paid" uses the paid 64K model under the ledger cap
# and bills/records the evals too; mode "free" stays on the free model.
#   LAB_PAID_MAX_USD=45 scripts/tinker/run_arm_live.sh <tag> paid [train_rl.py flags...]
#   scripts/tinker/run_arm_live.sh <tag> free [train_rl.py flags...]
set -u
TAG="$1"; MODE="$2"; shift 2
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/.inkling-abl/tinker_lab/.venv/bin/python"
LAB="$HOME/.inkling-abl/tinker_lab"
export LAB_JUDGE_MAX_USD="${LAB_JUDGE_MAX_USD:-6}"
PAIDFLAG=(); EVALRUN=""
if [ "$MODE" = paid ]; then
  : "${LAB_PAID_MAX_USD:?set LAB_PAID_MAX_USD}"
  export LAB_PAID_MAX_USD
  PAIDFLAG=(--paid); EVALRUN="${TAG}_evals"
fi
cd "$HERE" || exit 1
"$PY" train_rl.py --tag "$TAG" ${PAIDFLAG[@]+"${PAIDFLAG[@]}"} "$@" > "$LAB/logs/train_$TAG.stdout" 2>&1 &
TPID=$!
# Evaluate each checkpoint as soon as it is logged, without waiting for training to end.
seen=""
while true; do
  for line in $(grep '"event": "checkpoint"' "$LAB/logs/train_$TAG.log" 2>/dev/null | "$PY" -c '
import json, sys
for l in sys.stdin:
    r = json.loads(l)
    print("%s|%s" % (r["step"], r["sampler_path"]))'); do
    STEP="${line%%|*}"; SPATH="${line#*|}"
    case " $seen " in *" $STEP "*) ;; *)
      seen="$seen $STEP"
      LAB_PAID_RUN="$EVALRUN" nohup "$HERE/eval_ckpt_light.sh" "$TAG" "$STEP" "$SPATH" > /dev/null 2>&1 & ;;
    esac
  done
  kill -0 "$TPID" 2>/dev/null || break
  sleep 20
done
wait "$TPID"; echo "train exit $?" >> "$LAB/logs/train_$TAG.stdout"
echo "arm $TAG trained $(date '+%H:%M')" >> "$LAB/logs/train_$TAG.stdout"
