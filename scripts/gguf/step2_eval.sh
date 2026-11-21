#!/bin/bash
# ON POD, GPUs 6-7: base logits, KLD for every model, then generation/refusal checks for every model.
set -u
B=/data/base; U=/data/release/Inkling-2-Mini-Unbound
mkdir -p /data/release/gen
bash /root/gguf/kld_all.sh base
bash /root/gguf/kld_all.sh unbound_BF16=$U-BF16.gguf unbound_Q8_0=$U-Q8_0.gguf unbound_Q6_K=$U-Q6_K.gguf \
  unbound_Q5_K_M=$U-Q5_K_M.gguf unbound_Q4_K_M=$U-Q4_K_M.gguf unbound_IQ3_M=$U-IQ3_M.gguf unbound_IQ2_M=$U-IQ2_M.gguf \
  ggmlorg_Q8_0=$B/Inkling-2-Mini-Q8_0.gguf ggmlorg_IQ4_NL=$B/Inkling-2-Mini-IQ4_NL.gguf
gen() { python3 /root/gguf/eval_gen.py "$2" "$1" 6,7 8790 /data/release/gen "${3:-512}"; }
gen base_BF16 $B/Inkling-2-Mini-BF16.gguf 16
gen unbound_BF16 $U-BF16.gguf 16
for T in Q8_0 Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M; do gen unbound_$T $U-$T.gguf; done
gen base_Q8_0 $B/Inkling-2-Mini-Q8_0.gguf
echo STEP2_DONE
