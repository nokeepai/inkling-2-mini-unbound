#!/bin/bash
# ON an eval pod: wikitext KLD of a GGUF vs the original Q8_0 (same flags as compare/run_model.sh). Builds base.kld once.
#   bash kld.sh <name> <model.gguf> <gpu>
set -u
N=$1; M=$2; G=$3; B=/data/llama.cpp/build/bin; W=/data/compare/wiki.test.raw; K=/data/compare/kld; mkdir -p $K /root/v3/ev/$N
export CUDA_VISIBLE_DEVICES=$G
A=(-f $W -c 4096 --chunks 40 -ngl 99 -b 4096 -ub 512 --seed 1234)
if [ ! -f $K/base.done ]; then
  $B/llama-perplexity -m /data/base/Inkling-2-Mini-Q8_0.gguf "${A[@]}" --kl-divergence-base $K/base.kld > /root/v3/ev/kldbase.log 2>&1 && touch $K/base.done
fi
$B/llama-perplexity -m "$M" "${A[@]}" --kl-divergence-base $K/base.kld --kl-divergence > /root/v3/ev/$N/kld.log 2>&1
grep -aE 'Mean +KLD|Same top p|Mean +PPL' /root/v3/ev/$N/kld.log | tee /root/v3/ev/$N/kld.txt
