#!/bin/bash
# runs on pod after chain1: directions(e0) -> attribution (main conds) -> capture e0.9, e0.5 -> directions all -> extra attribution
cd /root/diag
until grep -q CHAIN1_DONE /root/logs/diag_chain1.log; do sleep 5; done
grep -q ALL_DONE /root/logs/diag_cap0.log || { echo "cap0 failed"; exit 1; }
python directions.py 0 > /root/logs/diag_dir0.log 2>&1 || { echo "dir0 failed"; exit 1; }
python attrib.py base attn_all mlp_all both_all both_0_5 both_6_17 both_18_23 > /root/logs/diag_attr.log 2>&1
python capture.py 0.9 harmful harmless > /root/logs/diag_cap09.log 2>&1
python directions.py 0 0.9 > /root/logs/diag_dir1.log 2>&1
python capture.py 0.5 harmful harmless > /root/logs/diag_cap05.log 2>&1
python directions.py 0 0.9 0.5 > /root/logs/diag_dir2.log 2>&1
python attrib.py attn_6_17 attn_18_23 attn_0_5 > /root/logs/diag_attr2.log 2>&1
echo CHAIN2_DONE
