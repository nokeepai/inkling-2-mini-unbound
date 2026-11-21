#!/usr/bin/env bash
# Start one Mac-side eval worker per GPU of a pod (agent `fleet`). Skips GPUs whose worker is already running.
#   bash ~/Code/inkling-abl/scripts/fleet/start_workers.sh ink-fleet-1 8      (single-GPU pod: N = 0 -> one worker, no --gpu)
# PIDs: scripts/fleet/pids/<pod>_g<i>.pid   Logs: state/agents/fleet_<pod>_g<i>.log   Stop: stop_workers.sh <pod>
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
POD=${1:?pod}
N=${2:?number of GPUs, 0 for a single-GPU pod}
mkdir -p $R/scripts/fleet/pids
start() {  # $1 = suffix, rest = extra args
  local sfx=$1; shift
  local pidf=$R/scripts/fleet/pids/${POD}${sfx}.pid
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "already running: $POD$sfx"; return; fi
  nohup python3 $R/harness/eval_worker.py --pod "$POD" "$@" > $R/state/agents/fleet_${POD}${sfx}.log 2>&1 &
  echo $! > "$pidf"
  echo "started $POD$sfx pid $!"
}
if [ "$N" = 0 ]; then
  start ""
else
  start _g0 --gpu 0
  sleep 20  # gpu 0 syncs the harness first; the others only read it
  for i in $(seq 1 $((N - 1))); do start _g$i --gpu $i; done
fi
