#!/bin/bash
# ON eval2: sealed-test harmful at answer cap 8192 for the other uncensored releases (same engine/flags as ours).
# OU = OpenUncensored's 20 wo_ud tensors patched into ggml-org BF16 -> Q8_0 (same as compare). XQ and AtomicChat
# AD-Q4_K are their published GGUFs. Each runs on 2 GPUs (test split into halves). HF_TOKEN from env.
set -u
M=/data/compare/models; mkdir -p $M/xq $M/atomic /data/cands/OU
cd /root/v11b
for k in 0 1; do mkdir -p thalf$k; chmod 700 thalf$k; done
python3 -c "
r=open('/root/v11b/test/harmful.jsonl').read().splitlines()
for k in range(2): open(f'/root/v11b/thalf{k}/harmful.jsonl','w').write(chr(10).join(r[k::2])+chr(10))"
( hf download IndrasMirror/Inkling-2-Mini-XQ-Uncensored Inkling-2-Mini-XQ-Uncensored.gguf README.md --local-dir $M/xq > /dev/null 2>&1
  for k in 0 1; do nohup bash harm8k.sh test8k_xq_h$k $((1+k)) $M/xq/Inkling-2-Mini-XQ-Uncensored.gguf /root/v11b/thalf$k > /root/logs/t8k_xq_h$k.log 2>&1 & done; wait; echo XQ_DONE ) &
( hf download AtomicChat/Inkling-2-Mini-Uncensored-GGUF Inkling-2-Mini-Uncensored-AD-Q4_K.gguf README.md --local-dir $M/atomic > /dev/null 2>&1
  for k in 0 1; do nohup bash harm8k.sh test8k_adq4_h$k $((3+k)) $M/atomic/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf /root/v11b/thalf$k > /root/logs/t8k_adq4_h$k.log 2>&1 & done; wait; echo ADQ4_DONE ) &
( hf download $SCRATCH_REPO cands/OU/tensors.safetensors --repo-type dataset --local-dir /data/hfs > /dev/null 2>&1; mv /data/hfs/cands/OU/tensors.safetensors /data/cands/OU/
  bash build_gguf.sh OU /data/cands/OU/tensors.safetensors > /root/logs/build_OU.log 2>&1; tail -n1 /root/logs/build_OU.log
  for k in 0 1; do nohup bash harm8k.sh test8k_ou_h$k $((5+k)) /data/v11/OU-Q8_0.gguf /root/v11b/thalf$k > /root/logs/t8k_ou_h$k.log 2>&1 & done; wait; echo OU_DONE ) &
wait; echo COMPETITORS_DONE
