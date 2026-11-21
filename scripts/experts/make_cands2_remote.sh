#!/bin/bash
# Pod-side round 2: stacks on mpoa's c1h_x4_ouprofile, plus lower-KL cond points on c1t002.
cd /root/exp
set -e
P=parents/c1h_x4_ouprofile
python build_router_cand.py $P cands/c4cd6_oupx4 32 6 cond
python build_router_cand.py $P cands/c4cd3_oupx4 32 3 cond
python build_router_cand.py $P cands/c4rp8_oupx4 8 0 rproj
python build_router_cand.py parents/c1t002 cands/c4cd6k8_c1t002 8 6 cond
python build_router_cand.py parents/c1t002 cands/c4cd6k16_c1t002 16 6 cond
for c in c4cd6_oupx4 c4cd3_oupx4 c4rp8_oupx4 c4cd6k8_c1t002 c4cd6k16_c1t002; do
  python /root/harness/build_cand.py cands/$c /root/w/orig /root/exp/built/$c | tail -1
  cp /root/exp/built/$c/sha256.txt cands/$c/sha256.txt
done
echo CANDS_DONE
