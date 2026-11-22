#!/bin/bash
# ON eval pod (CPU/disk; GPUs stay free for the running benchmark): build every v2 release artefact. HF_TOKEN from env.
#   1. v2 HF checkpoint = public v1.0 checkpoint with v2's 120 attention tensors spliced in place (splice_ckpt.py)
#   2. GGUF: patched ggml-org BF16 (120 tensors) -> Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ3_M, IQ2_M (ggml-org imatrix below
#      Q8_0; every edited attention tensor kept at q8_0), sha256 + KLD vs original Q8_0 for each (GPU 6)
#   3. Basalt v0.5.5 pack from the v2 checkpoint (same recipe as v1.0's pack)
set -u
R=/data/rel; mkdir -p $R/gguf $R/logs; cd /root/v11b
B=/data/llama.cpp/build/bin; N=Inkling-2-Mini-Unbound-v2
log() { echo "$(date -u +%T) $*" | tee -a $R/logs/status.txt; }
log START
( hf download nokeepai/Inkling-2-Mini-Unbound --revision f4afabb693 --local-dir $R/v1ck \
    --exclude "charts/*" "abliteration/*" "lora/*" "*.md" > $R/logs/dl_v1.log 2>&1; log DL_V1 rc=$? ) &
( hf download ggml-org/Inkling-2-Mini-GGUF imatrix.gguf --local-dir /data/base > $R/logs/dl_im.log 2>&1; log DL_IMATRIX rc=$? ) &
# GGUF BF16
python3 make_map_v11b.py /data/cands/E/tensors.safetensors $R/gguf/map.json && \
python3 patch_v11b.py /data/base/Inkling-2-Mini-BF16.gguf /data/orig /data/cands/E/tensors.safetensors $R/gguf/map.json \
  $R/gguf/$N-BF16.gguf "Inkling-2-Mini-Unbound-v2" > $R/logs/patch.log 2>&1
log PATCH_BF16 rc=$? "$(tail -n1 $R/logs/patch.log)"
wait
# quant ladder (CPU, parallel)
cp /data/v11/E-Q8_0.gguf $R/gguf/$N-Q8_0.gguf   # identical recipe (llama-quantize Q8_0 of the same patched BF16)
KEEP="--tensor-type attn_q=q8_0 --tensor-type attn_k=q8_0 --tensor-type attn_v=q8_0 --tensor-type attn_output=q8_0 --tensor-type attn_gate=q8_0"
for T in Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M; do
  CUDA_VISIBLE_DEVICES="" $B/llama-quantize --imatrix /data/base/imatrix.gguf $KEEP $R/gguf/$N-BF16.gguf $R/gguf/$N-$T.gguf $T 24 > $R/logs/q_$T.log 2>&1 &
done
# v2 HF checkpoint (CPU) in parallel with quantization
python3 splice_ckpt.py $R/v1ck /data/cands/E/tensors.safetensors $R/v2ck > $R/logs/splice.log 2>&1 || true
log SPLICE "$(tail -n1 $R/logs/splice.log)"
wait
for T in Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M; do log QUANT $T "$(tail -n1 $R/logs/q_$T.log | cut -c1-80)"; done
(cd $R/gguf && sha256sum *.gguf > SHA256SUMS && ls -l *.gguf > SIZES)
log GGUF_DONE
# KLD for every quant vs the original Q8_0 (GPU 6, sequential, same flags as compare)
for T in Q8_0 Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M; do
  (cd /data; CUDA_VISIBLE_DEVICES=6 $B/llama-perplexity -m $R/gguf/$N-$T.gguf -f /data/compare/wiki.test.raw -c 4096 --chunks 40 \
    -ngl 99 -b 4096 -ub 512 --seed 1234 --kl-divergence-base /data/compare/kld/base.kld --kl-divergence > $R/logs/kld_$T.log 2>&1)
  log KLD $T "$(grep -aE 'Mean +KLD' $R/logs/kld_$T.log)" "$(grep -aE 'Same top p' $R/logs/kld_$T.log)"
done
log KLD_DONE
# Basalt pack (v1.0 recipe: Basalt v0.5.5 converter, --checkpoint)
mkdir -p /root/b && cd /root/b
[ -d src ] || git clone -q --depth 1 --branch v0.5.5 https://github.com/jesdga95/basalt src
[ -d venv ] || { python3 -m venv venv && venv/bin/pip install -q -r src/requirements.txt numpy; }
cd src && venv_py=/root/b/venv/bin/python
$venv_py tools/convert/convert.py --checkpoint $R/v2ck --out $R/$N-NVFP4-q8.basalt --name "Inkling 2 Mini Unbound v2" \
  --workers 24 > $R/logs/basalt_convert.log 2>&1
log BASALT rc=$? "$(tail -n1 $R/logs/basalt_convert.log | cut -c1-100)"
(cd $R && sha256sum $N-NVFP4-q8.basalt > basalt.sha256)
log ALL_DONE
