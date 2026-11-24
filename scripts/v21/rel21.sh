#!/bin/bash
# ON ink-v21-rel (1x5090, 31 CPU, 93 GB RAM, 721 GB disk). Unattended v2.1 release build, same recipe as v2:
#   setup: stage31 bundle (scripts + sft1/dpo2 LoRAs), llama.cpp @19bc41dbb, ggml-org BF16/Q8_0/imatrix, orig attn,
#          wikitext, Unbound v2.0 HF checkpoint, Basalt v0.5.5 + official pack, base.kld
#   1. merge: v21 tensors = bf16(v2.0 + sft1:1.0 + dpo2:1.0)
#   2. HF checkpoint: splice the 120 tensors into the v2.0 checkpoint in place (every other byte verified identical)
#   3. GGUF: patched BF16 (mapping proved vs orig) -> Q8_0 + Q6_K/Q5_K_M/Q4_K_M/IQ3_M/IQ2_M (imatrix, attn kept q8_0)
#      sha256 + wikitext KLD vs original Q8_0 for every quant; Q8_0 must equal the evaluated v31C file's KLD
#   4. Basalt v0.5.5 pack + verify_basalt_v2.py against the official pack
#   5. r16/r64 SVD LoRAs (PEFT + GGUF) + KLD validation vs the full v2.1 Q8_0
#   6. finetune/ package (stacked LoRA on top of v2.0, reconstruction checked)
#   7. load + short-generation smoke test of every GGUF (lengths only, no text logged)
# Progress: /root/logs/rel21.log. HF token must be in $HF_HOME/token (pods set HF_HOME=/workspace/hf).
set -u
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p /root/logs; L=/root/logs/rel21.log
log() { echo "$(date -u +%T) $*" >> $L; }
R=/data/rel; N=Inkling-2-Mini-Unbound-v2.1; B=/data/llama.cpp/build/bin
log START "$(df -h /data / | tail -n +2 | tr -s ' ' | cut -d' ' -f4,6 | tr '\n' ' ')"

# ---- setup ----
mkdir -p /root/stage /root/v11b /root/compare/bin /root/v3 /root/harness $R/gguf $R/logs
tar xzf /root/stage31.tgz -C /root/stage
cp -r /root/stage/v11b/* /root/v11b/; cp -r /root/stage/compare_bin/* /root/compare/bin/ 2>/dev/null
cp -r /root/stage/v3/* /root/v3/; cp /root/stage/harness/* /root/harness/ 2>/dev/null
cp /root/v21/*.py /root/v11b/
bash /root/v11b/eval_setup.sh_v3 > /root/logs/setup_fast.log 2>&1; log FAST "$(tail -n1 /root/logs/setup_fast.log)"
( hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-BF16.gguf imatrix.gguf --local-dir /data/base > $R/logs/dl_bf16.log 2>&1; log DL_BF16_IMATRIX rc=$? ) &
( hf download nokeepai/Inkling-2-Mini-Unbound chat_template.jinja config.json generation_config.json hf_quant_config.json \
    model-00001-of-00001.safetensors model.safetensors.index.json mtp.safetensors processor_config.json \
    special_tokens_map.json tiktoken/tokenizer.model tokenizer.json tokenizer_config.json \
    --revision v2.0 --local-dir /data/v2hf > $R/logs/dl_v2hf.log 2>&1; log DL_V2HF rc=$? ) &
( mkdir -p /root/b && cd /root/b && git clone -q --depth 1 --branch v0.5.5 https://github.com/jesdga95/basalt src \
    && python3 -m venv venv && venv/bin/pip install -q -r src/requirements.txt numpy > $R/logs/basalt_venv.log 2>&1
  hf download jesdga/Inkling-2-Mini-Basalt Inkling-2-Mini-NVFP4-DQ8.basalt --local-dir /root/b/official > $R/logs/dl_off.log 2>&1
  log BASALT_TOOLS rc=$? "$(cd /root/b/src && git rev-parse --short HEAD)" ) &
( cd /data && CUDA_VISIBLE_DEVICES=0 $B/llama-perplexity -m /data/base/Inkling-2-Mini-Q8_0.gguf -f /data/compare/wiki.test.raw -c 4096 \
    --chunks 40 -ngl 99 -b 4096 -ub 512 --seed 1234 --kl-divergence-base /data/compare/kld/base.kld > $R/logs/kldbase.log 2>&1
  log BASE_KLD rc=$? ) &
wait
log SETUP_DONE

# ---- 1. merge ----
cd /root/v11b
python3 merge_stack.py /data/v2hf hf_name_map.json $R/v21_tensors.safetensors /root/v3/loras/sft1.pt:1.0 /root/v3/loras/dpo2.pt:1.0 \
  > $R/logs/merge.log 2>&1; log MERGE rc=$? "$(tail -n1 $R/logs/merge.log)"
sha256sum $R/v21_tensors.safetensors >> $L

# ---- 2. HF checkpoint ----
python3 splice_ckpt.py /data/v2hf $R/v21_tensors.safetensors $R/v21ck > $R/logs/splice.log 2>&1
log SPLICE rc=$? "$(tail -n1 $R/logs/splice.log)"

# ---- 3. GGUF ladder ----
python3 make_map_v11b.py $R/v21_tensors.safetensors $R/gguf/map.json && \
python3 patch_v11b.py /data/base/Inkling-2-Mini-BF16.gguf /data/orig $R/v21_tensors.safetensors $R/gguf/map.json \
  $R/gguf/$N-BF16.gguf "$N" > $R/logs/patch.log 2>&1
log PATCH_BF16 rc=$? "$(tail -n1 $R/logs/patch.log)"
$B/llama-quantize $R/gguf/$N-BF16.gguf $R/gguf/$N-Q8_0.gguf Q8_0 > $R/logs/q_Q8_0.log 2>&1; log QUANT Q8_0 rc=$?
KEEP="--tensor-type attn_q=q8_0 --tensor-type attn_k=q8_0 --tensor-type attn_v=q8_0 --tensor-type attn_output=q8_0 --tensor-type attn_gate=q8_0"
for T in Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M; do
  CUDA_VISIBLE_DEVICES="" $B/llama-quantize --imatrix /data/base/imatrix.gguf $KEEP $R/gguf/$N-BF16.gguf $R/gguf/$N-$T.gguf $T 6 \
    > $R/logs/q_$T.log 2>&1 &
done
wait
for T in Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M; do log QUANT $T "$(tail -n1 $R/logs/q_$T.log | cut -c1-80)"; done
(cd $R/gguf && sha256sum $N-Q*.gguf $N-IQ*.gguf > SHA256SUMS && ls -l $N-*.gguf > SIZES)
log GGUF_DONE

# ---- 4. Basalt (CPU) in parallel with KLD (GPU) ----
( cd /root/b/src && /root/b/venv/bin/python tools/convert/convert.py --checkpoint $R/v21ck --out $R/$N-NVFP4-q8.basalt \
    --name "Inkling 2 Mini Unbound v2.1" --workers 24 > $R/logs/basalt.log 2>&1
  log BASALT rc=$? "$(tail -n1 $R/logs/basalt.log | cut -c1-100)"
  (cd $R && sha256sum $N-NVFP4-q8.basalt > basalt.sha256)
  cd /root/b/src/tools/convert && /root/b/venv/bin/python /root/v11b/verify_basalt_v2.py $R/$N-NVFP4-q8.basalt \
    /root/b/official/Inkling-2-Mini-NVFP4-DQ8.basalt $R/v21ck > $R/logs/basalt_verify.log 2>&1
  log BASALT_VERIFY rc=$? "$(tail -n1 $R/logs/basalt_verify.log | cut -c1-100)" ) &
for T in Q8_0 Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M; do
  (cd /data; CUDA_VISIBLE_DEVICES=0 $B/llama-perplexity -m $R/gguf/$N-$T.gguf -f /data/compare/wiki.test.raw -c 4096 --chunks 40 \
    -ngl 99 -b 4096 -ub 512 --seed 1234 --kl-divergence-base /data/compare/kld/base.kld --kl-divergence > $R/logs/kld_$T.log 2>&1)
  log KLD $T "$(grep -aE 'Mean +KLD' $R/logs/kld_$T.log)" "$(grep -aE 'Same top p' $R/logs/kld_$T.log)"
done
log KLD_DONE

# ---- 5. SVD LoRAs + validation ----
python3 lora_build_v21.py > $R/logs/lora_build.log 2>&1; log LORA_BUILD rc=$?
LV=$R/lora_v21; K=$R/kld_v21base; mkdir -p $K
(cd /data; CUDA_VISIBLE_DEVICES=0 $B/llama-perplexity -m $R/gguf/$N-Q8_0.gguf -f /data/compare/wiki.test.raw -c 4096 --chunks 40 \
  -ngl 99 -b 4096 -ub 512 --seed 1234 --kl-divergence-base $K/v21.kld > $R/logs/kld_v21base.log 2>&1)
for r in 16 64; do
  (cd /data; CUDA_VISIBLE_DEVICES=0 $B/llama-perplexity -m /data/base/Inkling-2-Mini-Q8_0.gguf --lora $LV/$N-LoRA-r$r.gguf \
    -f /data/compare/wiki.test.raw -c 4096 --chunks 40 -ngl 99 -b 4096 -ub 512 --seed 1234 --kl-divergence-base $K/v21.kld \
    --kl-divergence > $R/logs/kld_lora_r$r.log 2>&1)
  log LORA_KLD r$r "$(grep -aE 'Mean +KLD' $R/logs/kld_lora_r$r.log)" "$(grep -aE 'Same top p' $R/logs/kld_lora_r$r.log)"
done

# ---- 6. finetune/ package ----
grep "$N-Q8_0.gguf" $R/gguf/SHA256SUMS > $R/gguf/q8.sha256
python3 package_v21.py /data/v2hf hf_name_map.json $R/v21_tensors.safetensors $R/gguf/q8.sha256 \
  $R/finetune /root/v3/loras/sft1.pt /root/v3/loras/dpo2.pt > $R/logs/package.log 2>&1
log PACKAGE rc=$? "$(tail -n1 $R/logs/package.log | cut -c1-200)"

# ---- 7. smoke test every GGUF ----
bash /root/v21/smoke.sh $R/gguf > $R/logs/smoke.log 2>&1; log SMOKE rc=$?; grep SMOKE $R/logs/smoke.log >> $L
du -sh $R/* >> $L
log ALL_DONE
