#!/bin/bash
# Build C4 router candidates on ink-research-3 (stacked on c1t002 and c0_lr), test build_cand, pull back.
R=~/Code/inkling-abl
S="bash $R/scripts/ink_lium.sh"
P=ink-research-3
$S exec $P "mkdir -p /root/exp/parents/c1t002 /root/exp/parents/c0_lr /root/exp/cands /root/harness" >/dev/null
$S scp $P $R/methods/mpoa/cands/c1t002/lowrank.safetensors /root/exp/parents/c1t002/lowrank.safetensors >/dev/null
$S scp $P $R/methods/baselines/cands/c0_lr/lowrank.safetensors /root/exp/parents/c0_lr/lowrank.safetensors >/dev/null
$S scp $P $R/scripts/experts/build_router_cand.py /root/exp/build_router_cand.py >/dev/null
$S scp $P $R/harness/build_cand.py /root/harness/build_cand.py >/dev/null
$S scp $P $R/scripts/experts/make_cands_remote.sh /root/exp/make_cands_remote.sh >/dev/null
$S exec $P "bash /root/exp/make_cands_remote.sh"
for c in c4rp8_c1t002 c4cd6_c1t002 c4cd3_c1t002 c4rp8_c0lr c4cd6_c0lr; do
  mkdir -p $R/methods/experts/cands/$c
  for f in lowrank.safetensors spec.json sha256.txt; do
    $S scp $P /root/exp/cands/$c/$f $R/methods/experts/cands/$c/$f --download >/dev/null
  done
done
ls -la $R/methods/experts/cands/*
