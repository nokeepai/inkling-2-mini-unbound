#!/usr/bin/env bash
# Stop the Mac-side eval workers this folder started for a pod (only PIDs recorded in scripts/fleet/pids/).
# A job already running on the pod finishes and restores orig on its own; its claim reopens after 2 h
# (or append a release row, see README).
#   bash ~/Code/inkling-abl/scripts/fleet/stop_workers.sh ink-fleet-1
R=/Users/nikshepsvn/Code/inkling-abl
POD=${1:?pod}
for f in $R/scripts/fleet/pids/${POD}*.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then kill "$pid" && echo "stopped $(basename "$f" .pid) ($pid)"; fi
  rm -f "$f"
done
