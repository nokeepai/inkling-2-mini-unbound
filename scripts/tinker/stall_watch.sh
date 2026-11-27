#!/bin/bash
# Exits (and so wakes the orchestrator) when nothing under the lab dir has changed for STALL_MIN minutes.
# Why: a subagent once sat idle 40 min waiting for a notification that never came.
LAB="$HOME/.inkling-abl/tinker_lab"
STALL_MIN="${1:-15}"
# Grace period: give a freshly briefed agent time to start writing before judging it stalled.
sleep $((STALL_MIN * 60))
while true; do
  newest=$(find "$LAB" -type f -newermt "-${STALL_MIN} minutes" -print -quit 2>/dev/null)
  if [ -z "$newest" ]; then
    echo "STALL: no file under $LAB changed in ${STALL_MIN} min at $(date '+%H:%M')"
    ls -lt "$LAB" "$LAB/logs" | head -12
    exit 0
  fi
  sleep 120
done
