#!/bin/bash
# Remove a pod, recording its final spend first. Usage: pod_rm.sh <pod-name> [rating 1-5]
# Safety: the Lium account also runs nokeep production pods. Only exact names
# starting with "ink-" are accepted; never "all", -a, ps row numbers, ids or huids.
set -u
export LIUM_NONINTERACTIVE=1
export LIUM_NO_POD_INDEX=1
R=~/Code/inkling-abl
NAME="${1:-}"
source "$R/scripts/ink_guard.sh"
ink_guard "$NAME" || exit 2
SPENT=$(lium ps --format json | python3 -c "import sys,json;print(next((p['spent_usd'] for p in json.load(sys.stdin) if p['name']=='$NAME'),0))")
echo "{\"time\":\"$(date -u +%FT%TZ)\",\"name\":\"$NAME\",\"final_spent_usd\":$SPENT}" >> "$R/state/pods_removed.jsonl"
if [ -n "${2:-}" ]; then
  lium rm "$NAME" --name-only --rating "$2" -y --format json
else
  lium rm "$NAME" --name-only -y --format json
fi
python3 "$R/scripts/ledger.py" > /dev/null
