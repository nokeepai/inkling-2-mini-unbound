#!/bin/bash
# Local: wait until a pod log contains a pattern (or FAILED), polling every 60 s; also refresh the status line.
#   bash wait_log.sh <pod log path> <pattern> <status text> [max_minutes]
R=~/Code/inkling-abl
LOG=$1; PAT=$2; TXT=$3; MAX=${4:-120}
for i in $(seq 1 $MAX); do
  OUT=$(bash $R/scripts/ink_lium.sh exec ink-suite-lc "tail -n 4 $LOG" 2>/dev/null)
  PROG=$(echo "$OUT" | grep -E '^[0-9]+/[0-9]+ ' | tail -1)
  echo "$(date -u +%H:%MZ) ruler: $TXT; progress ${PROG:-n/a}" > $R/state/agents/suite_ruler.txt
  if echo "$OUT" | grep -qE "$PAT|FAILED"; then echo "$OUT"; exit 0; fi
  sleep 60
done
echo "TIMEOUT"; echo "$OUT"
