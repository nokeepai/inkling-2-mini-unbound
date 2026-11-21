#!/bin/bash
L="bash /Users/nikshepsvn/Code/inkling-abl/scripts/ink_lium.sh"
$L exec ink-gguf-1 "mkdir -p /data/compare/ou"
$L scp ink-gguf-1 /Users/nikshepsvn/Code/inkling-abl/methods/baselines/cands/ou/tensors.safetensors /data/compare/ou/tensors.safetensors
$L scp ink-gguf-1 /Users/nikshepsvn/Code/inkling-abl/scripts/compare/patch_gguf_ou.py /root/compare/bin/patch_gguf_ou.py
