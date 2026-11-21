#!/bin/bash
# Start weight fetch + image pull on the 8x5090 fleet pod.
export LIUM_NONINTERACTIVE=1
R=~/Code/inkling-abl
lium scp ink-fleet-1 "$R/scripts/pod_fetch.sh" /root/pod_fetch.sh 2>&1 | tail -1
lium exec ink-fleet-1 "nohup bash /root/pod_fetch.sh > /root/fetch.log 2>&1 &" 2>&1 | tail -1
lium exec ink-fleet-1 "nohup docker pull lmsysorg/sglang:inkling-2-mini > /root/pull.log 2>&1 &" 2>&1 | tail -1
lium exec ink-fleet-1 "nvidia-smi -L; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1; df -h /root | tail -1; nproc" 2>&1 | tail -12
python3 $R/scripts/ledger.py | sed -n 3,6p
