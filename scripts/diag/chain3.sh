#!/bin/bash
# cut-down chain: capture e0 (384+384 + atoms) -> directions e0 -> attribution -> capture e0.9 (256+256) -> directions
cd /root/diag
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LIMIT=384 python capture.py 0 harmful harmless math code honest crisis > /root/logs/diag_cap0.log 2>&1
grep -q ALL_DONE /root/logs/diag_cap0.log || { echo "cap0 failed"; exit 1; }
python directions.py 0 > /root/logs/diag_dir0.log 2>&1 || { echo "dir0 failed"; exit 1; }
echo DIR0_DONE
python attrib.py > /root/logs/diag_attr.log 2>&1
echo ATTR_DONE
LIMIT=256 python capture.py 0.9 harmful harmless > /root/logs/diag_cap09.log 2>&1
python directions.py 0 0.9 > /root/logs/diag_dir1.log 2>&1
echo CHAIN3_DONE
