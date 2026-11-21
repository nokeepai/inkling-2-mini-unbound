#!/bin/bash
# Quick state of the suite pods.
R=~/Code/inkling-abl
for p in ${@:-ink-suite-1 ink-suite-lc}; do
  echo "=== $p"
  bash $R/scripts/ink_lium.sh exec $p "date -u; tail -n 2 /root/fetch.log; tail -n 2 /root/pull.log; du -sh /root/w/orig /root/w/ou 2>/dev/null; df -h /srv /root | tail -2; nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader | sort | uniq -c; docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}'; docker ps --format '{{.Names}} {{.Status}}'; python3 -c 'import torch,transformers;print(torch.__version__,transformers.__version__)' 2>&1 | tail -1; nproc; free -g | head -2; ls /srv; tail -n 3 /root/logs/*.log 2>/dev/null | tail -20" 2>&1
done
