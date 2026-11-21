#!/bin/bash
# Control machine: push fetch script to both Phase 0 pods, start downloads + sglang pull, print pod facts.
export LIUM_NONINTERACTIVE=1
R=~/Code/inkling-abl
for p in ink-research-1 ink-eval-1; do
  lium scp "$p" "$R/scripts/pod_fetch.sh" /root/pod_fetch.sh 2>&1 | tail -1
  lium exec "$p" "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h /root | tail -1; free -g | sed -n 2p; docker --version" 2>&1 | tail -5
  lium exec "$p" "nohup bash /root/pod_fetch.sh > /root/fetch.log 2>&1 &" 2>&1 | tail -1
done
lium exec ink-eval-1 "nohup docker pull lmsysorg/sglang:inkling-2-mini > /root/pull.log 2>&1 &" 2>&1 | tail -1
lium exec ink-research-1 "nohup docker pull lmsysorg/sglang:inkling-2-mini > /root/pull.log 2>&1 &" 2>&1 | tail -1
echo KICKOFF_DONE
