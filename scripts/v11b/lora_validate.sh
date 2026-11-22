#!/bin/bash
# ON eval pod: validate v2 LoRAs. (1) KLD of original Q8_0 + LoRA vs full v2 Q8_0 (wikitext, same flags as compare).
# (2) behaviour: original + r64 LoRA on the first 20 sealed-test harmful prompts, thinking off (refusal now, Jev later).
set -u
B=/data/llama.cpp/build/bin; W=/data/compare/wiki.test.raw; L=/data/rel/lora_v2; K=/data/rel/kld_v2base
mkdir -p $K; cd /data
CUDA_VISIBLE_DEVICES=0 $B/llama-perplexity -m /data/v11/E-Q8_0.gguf -f $W -c 4096 --chunks 40 -ngl 99 -b 4096 -ub 512 --seed 1234 \
  --kl-divergence-base $K/v2.kld > /data/rel/logs/kld_v2base.log 2>&1
for r in 16 64; do
  CUDA_VISIBLE_DEVICES=$([ $r = 16 ] && echo 0 || echo 1) $B/llama-perplexity -m /data/base/Inkling-2-Mini-Q8_0.gguf --lora $L/Inkling-2-Mini-Unbound-v2-LoRA-r$r.gguf \
    -f $W -c 4096 --chunks 40 -ngl 99 -b 4096 -ub 512 --seed 1234 --kl-divergence-base $K/v2.kld --kl-divergence > /data/rel/logs/kld_lora_r$r.log 2>&1 &
done
( export CMP_DATA=/root/v11b/test CMP_TOK=/data/compare/models/tok CMP_METRICS=harmful CMP_ANS_CAP=8192 CMP_SKIP_KLD=1 CMP_LIMIT=20 CMP_CTX=49152 CMP_NP=6
  cd /root/compare/bin && bash run_model.sh lora64_check 2 8202 /data/base/Inkling-2-Mini-Q8_0.gguf $L/Inkling-2-Mini-Unbound-v2-LoRA-r64.gguf ) &
wait
for r in 16 64; do echo "LoRA r$r vs full v2: $(grep -aE 'Mean +KLD' /data/rel/logs/kld_lora_r$r.log) | $(grep -aE 'Same top p' /data/rel/logs/kld_lora_r$r.log)"; done
echo VALIDATE_DONE
