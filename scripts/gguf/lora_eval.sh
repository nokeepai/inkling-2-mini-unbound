#!/bin/bash
# ON POD, GPU 6: LoRA r16/r64 on ggml-org Q8_0 -- KLD vs patched Unbound Q8_0 and vs base BF16, plus refusal check.
set -u
export CUDA_VISIBLE_DEVICES=6
P=/data/llama.cpp/build-m/bin/llama-perplexity
Q8=/data/base/Inkling-2-Mini-Q8_0.gguf
UQ8=/data/release/Inkling-2-Mini-Unbound-Q8_0.gguf
A="-f /data/work/wiki.test.raw -c 512 -b 512 -ub 16 --chunks 40"
K=/data/release/kld
L=/data/release/Inkling-2-Mini-Unbound-LoRA
# reference logits of the full edit (our patched Q8_0)
$P -m $UQ8 $A --kl-divergence-base /data/work/unbound_q8.kld > $K/ref_unbound_q8.log 2>&1
echo "ref rc=$?"
kl() { tag=$1; base=$2; shift 2; $P $A --kl-divergence-base $base --kl-divergence "$@" > $K/$tag.log 2>&1
  echo "$tag rc=$? $(grep -E '^Mean +KLD' $K/$tag.log) | $(grep -E '^Same top p' $K/$tag.log)"; }
kl vsUQ8_ggmlQ8_nolora /data/work/unbound_q8.kld -m $Q8
for r in 16 64; do
  kl vsUQ8_lora_r$r /data/work/unbound_q8.kld -m $Q8 --lora $L-r$r.gguf
  kl vsBF16_lora_r$r /data/work/base_bf16.kld -m $Q8 --lora $L-r$r.gguf
done
for r in 16 64; do
  EXTRA_ARGS="--lora $L-r$r.gguf" python3 /root/gguf/eval_gen.py $Q8 lora_r$r 6 8791 /data/release/gen 2>/dev/null
done
echo LORA_EVAL_DONE
