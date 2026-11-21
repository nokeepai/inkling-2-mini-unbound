#!/bin/bash
R=/Users/nikshepsvn/Code/inkling-abl
L="bash $R/scripts/ink_lium.sh"
$L exec ink-gguf-1 "mkdir -p /root/gguf /data/logs"
for f in pod_setup.sh; do $L scp ink-gguf-1 $R/scripts/gguf/$f /root/gguf/$f; done
$L scp ink-gguf-1 $R/harness/build_cand.py /root/gguf/build_cand.py
$L scp ink-gguf-1 $R/methods/mpoa/cands/c1h_r1_svd1_s3_L4/lowrank.safetensors /root/gguf/cand/lowrank.safetensors
$L scp ink-gguf-1 $R/methods/mpoa/cands/c1h_r1_svd1_s3_L4/sha256.txt /root/gguf/candbuild/sha256.txt
$L scp ink-gguf-1 $R/out/final/edit.pt /root/gguf/edit.pt
$L scp ink-gguf-1 $R/harness/refusal_phrases.txt /root/gguf/refusal_phrases.txt
$L scp ink-gguf-1 $R/data/val/harmful.jsonl /root/gguf/harmful.jsonl
$L exec ink-gguf-1 "nohup bash /root/gguf/pod_setup.sh > /data/logs/setup.log 2>&1 &"
$L exec ink-gguf-1 "ls -la /root/gguf /root/gguf/cand /root/gguf/candbuild"
