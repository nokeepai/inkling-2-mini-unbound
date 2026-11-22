#!/bin/bash
# ON a fresh eval pod (replacement after the 04:12Z TTL loss). HF_TOKEN from env (private scratch repo).
# Rebuilds: llama.cpp @19bc41dbb (tarball from scratch repo), ggml-org BF16+Q8_0 GGUFs, v1.0 Q8_0, orig attention
# tensors, tokenizer, wikitext, compare/bin. Then base KLD (orig Q8_0) on GPU 0.
set -u
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /data/base /data/orig /data/v10 /data/v11 /data/logs /data/hfs /data/compare/models/tok /data/compare/kld /root/compare/logs /root/compare/out /root/logs
pip install -q transformers safetensors numpy pyarrow hf_transfer sentencepiece 2>&1 | tail -1
( hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-BF16.gguf Inkling-2-Mini-Q8_0.gguf --local-dir /data/base > /data/logs/dl_base.log 2>&1; echo "DL_BASE rc=$?" ) &
( hf download nokeepai/Inkling-2-Mini-Unbound-GGUF Inkling-2-Mini-Unbound-Q8_0.gguf --local-dir /data/v10 > /data/logs/dl_v10.log 2>&1; echo "DL_V10 rc=$?" ) &
( hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da --include "tokenizer*" "*.jinja" "special_tokens_map.json" "config.json" "generation_config.json" "chat_template*" --local-dir /data/compare/models/tok > /dev/null 2>&1
  hf download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/test-*" --local-dir /data/compare/models/wikitext > /dev/null 2>&1
  python3 /root/v11b/prep_wiki.py; echo "DL_SMALL done" ) &
hf download $SCRATCH_REPO llama_19bc.tar.gz --repo-type dataset --local-dir /data/hfs > /dev/null 2>&1
hf download $SCRATCH_REPO --include 'orig_attn/*' --repo-type dataset --local-dir /data/hfs > /dev/null 2>&1
cp /data/hfs/orig_attn/* /data/orig/; ls /data/orig | wc -l
(apt-get install -y -q cmake > /dev/null 2>&1 || pip install -q cmake)
bash /root/v11b/eval_build.sh /data/hfs/llama_19bc.tar.gz
wait
ls -la /data/base /data/v10 /data/compare/wiki.test.raw
echo SETUP_DONE
# base KLD for the screens (same as compare --base-kld-first, KLD only)
cd /data && CUDA_VISIBLE_DEVICES=0 /data/llama.cpp/build/bin/llama-perplexity -m /data/base/Inkling-2-Mini-Q8_0.gguf -f /data/compare/wiki.test.raw -c 4096 --chunks 40 -ngl 99 -b 4096 -ub 512 --seed 1234 --kl-divergence-base /data/compare/kld/base.kld > /root/compare/logs/kldbase.log 2>&1 && touch /data/compare/kld/base.done
echo BASEKLD_DONE
