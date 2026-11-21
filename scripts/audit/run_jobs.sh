#!/bin/bash
# Pod-side (SGLang pod): run sealed_run.py jobs in parallel, one per GPU server.
# Usage: bash run_jobs.sh <data_dir> <split> <robust_dir> <think_cap> <jobs_file>
#   jobs_file lines: "<gpu|-> <cand> <blocks> <tag> [limit]"   (gpu "-" = single-GPU pod, container sgl on port 3001)
# Logs: /root/logs/audit_<cand>_<tag>.log ; outputs /srv/audit/evals/<cand>_<tag>/
set -uo pipefail
DATA=$1; SPLIT=$2; ROB=$3; CAP=$4; JOBS=$5
cd /srv/audit/scripts
pids=()
while read -r gpu cand blocks tag lim; do
  [ -z "${gpu:-}" ] && continue
  case "$gpu" in \#*) continue ;; esac
  GARG=""; SLOT=""
  if [ "$gpu" != "-" ]; then GARG="--gpu $gpu"; SLOT="__g$gpu"; fi
  if [ "$cand" != "orig" ]; then
    rm -rf "/srv/cands/${cand}${SLOT}"; cp -r "/srv/audit/cands/$cand" "/srv/cands/${cand}${SLOT}"
  fi
  LARG=""; [ -n "${lim:-}" ] && [ "$lim" != "0" ] && LARG="--limit $lim"
  nohup python3 sealed_run.py --cand "$cand" --data "$DATA" --split "$SPLIT" $GARG --blocks "$blocks" \
    --robust "$ROB" --think-cap "$CAP" --tag "$tag" $LARG > "/root/logs/audit_${cand}_${tag}.log" 2>&1 &
  pids+=($!)
  echo "started $cand $tag on gpu $gpu pid $! $(date -u +%H:%M:%S)"
  if [ "$gpu" = "-" ]; then wait $! || echo "FAILED $cand $tag"; continue; fi   # one server: run serially
  sleep 20   # stagger the 19 GB merge copies
done < "$JOBS"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "RUN_JOBS_DONE failed=$fail"
