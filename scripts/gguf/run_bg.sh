#!/bin/bash
# ON POD: run_bg.sh <script> <log> [args...]  -- detach a pod script with nohup + setsid.
S=$1; LOG=$2; shift 2
setsid nohup bash "$S" "$@" > "$LOG" 2>&1 < /dev/null &
echo "started $S pid $!"
