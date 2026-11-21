#!/bin/bash
# runs on pod: wait for free GPU, then leak, then capture effort 0
cd /root/diag
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 4000 ]; do sleep 10; done
echo "GPU free $(date -u)"
python leak.py > /root/logs/diag_leak.log 2>&1
python capture.py 0 harmful harmless math code honest crisis > /root/logs/diag_cap0.log 2>&1
echo CHAIN1_DONE
