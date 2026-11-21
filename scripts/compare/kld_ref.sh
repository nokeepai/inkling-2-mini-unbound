#!/bin/bash
# ON pod: KLD-only reference rows (4-bit quant noise floor) on GPU $1, same flags as run_model.sh's kld().
GPU=$1; B=/data/llama.cpp/build/bin; KLD=/data/compare/kld; LOG=/root/compare/logs
export CUDA_VISIBLE_DEVICES=$GPU
for pair in "orig_iq4nl:/data/base/Inkling-2-Mini-IQ4_NL.gguf" "unbound_q4km:/data/release/Inkling-2-Mini-Unbound-Q4_K_M.gguf"; do
  n=${pair%%:*}; m=${pair#*:}
  $B/llama-perplexity -m $m -f /data/compare/wiki.test.raw -c 4096 --chunks 40 -ngl 99 -b 4096 -ub 512 --seed 1234 \
    --kl-divergence-base $KLD/base.kld --kl-divergence > $LOG/$n.kld.log 2>&1
  echo "$(date -u +%FT%TZ) $n KLD rc=$? $(ls -la $m)" >> $LOG/kld_ref.status
done
echo "$(date -u +%FT%TZ) ALL_DONE" >> $LOG/kld_ref.status
