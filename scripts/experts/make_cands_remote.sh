#!/bin/bash
# Pod-side: build the 5 C4 candidates and check build_cand.py accepts and builds each (sha256.txt = built sha).
cd /root/exp
set -e
python build_router_cand.py parents/c1t002 cands/c4rp8_c1t002 8 0 rproj
python build_router_cand.py parents/c1t002 cands/c4cd6_c1t002 32 6 cond
python build_router_cand.py parents/c1t002 cands/c4cd3_c1t002 32 3 cond
python build_router_cand.py parents/c0_lr cands/c4rp8_c0lr 8 0 rproj
python build_router_cand.py parents/c0_lr cands/c4cd6_c0lr 32 6 cond
for c in c4rp8_c1t002 c4cd6_c1t002 c4cd3_c1t002 c4rp8_c0lr c4cd6_c0lr; do
  python /root/harness/build_cand.py cands/$c /root/w/orig /root/exp/built/$c | tail -1
  cp /root/exp/built/$c/sha256.txt cands/$c/sha256.txt
done
echo CANDS_DONE
