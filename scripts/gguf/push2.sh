#!/bin/bash
R=/Users/nikshepsvn/Code/inkling-abl
L="bash $R/scripts/ink_lium.sh"
$L exec ink-gguf-1 "mkdir -p /root/gguf/cand /root/gguf/candbuild"
$L scp ink-gguf-1 $R/methods/mpoa/cands/c1h_r1_svd1_s3_L4/lowrank.safetensors /root/gguf/cand/lowrank.safetensors
$L scp ink-gguf-1 $R/methods/mpoa/cands/c1h_r1_svd1_s3_L4/sha256.txt /root/gguf/candbuild/sha256.txt
$L exec ink-gguf-1 "ls -la /root/gguf/cand /root/gguf/candbuild; tail -5 /data/logs/setup.log; tail -c 300 /data/logs/dl_base.log"
