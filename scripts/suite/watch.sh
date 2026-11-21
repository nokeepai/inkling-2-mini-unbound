#!/usr/bin/env bash
# Mac: every 3 min, print new done/score/ERROR/restart events from the suite workers.
R=/Users/nikshepsvn/Code/inkling-abl
seen=$R/suite/.watch_seen
touch $seen
while true; do
  bash $R/scripts/ink_lium.sh exec ink-suite-1 "cat /root/logs/worker_*.events" 2>/dev/null \
    | grep -E 'done |score |ERROR|restart|WARNING|already' | sort > $seen.new
  comm -13 $seen $seen.new
  mv $seen.new $seen
  sleep 180
done
