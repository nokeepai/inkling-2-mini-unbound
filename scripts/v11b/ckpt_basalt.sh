#!/bin/bash
# ON eval pod: v2 HF checkpoint (splice into the v1.0 TAG's checkpoint) + Basalt v0.5.5 pack. Separate paths from
# release_build.sh (whose own splice/Basalt steps fail harmlessly: its v1 download was wrong). HF_TOKEN from env.
set -u
R=/data/rel; C=$R/ckb; mkdir -p $C $R/logs; cd /root/v11b
log() { echo "$(date -u +%T) $*" | tee -a $R/logs/status_ckb.txt; }
log START
hf download nokeepai/Inkling-2-Mini-Unbound chat_template.jinja config.json generation_config.json hf_quant_config.json \
  model-00001-of-00001.safetensors model.safetensors.index.json mtp.safetensors processor_config.json \
  special_tokens_map.json tiktoken/tokenizer.model tokenizer.json tokenizer_config.json \
  --revision v1.0 --local-dir $C/v1ck > $R/logs/dl_v1b.log 2>&1
log DL_V1 rc=$? "$(du -sh $C/v1ck | cut -f1)"
sha256sum $C/v1ck/model-00001-of-00001.safetensors | tee -a $R/logs/status_ckb.txt
python3 splice_ckpt.py $C/v1ck /data/cands/E/tensors.safetensors $C/v2ck > $R/logs/splice_b.log 2>&1
log SPLICE rc=$? "$(tail -n1 $R/logs/splice_b.log)"
mkdir -p /root/b && cd /root/b
[ -d src ] || git clone -q --depth 1 --branch v0.5.5 https://github.com/jesdga95/basalt src
[ -d venv ] || { python3 -m venv venv && venv/bin/pip install -q -r src/requirements.txt numpy; }
log BASALT_TOOLS "$(cd src && git rev-parse --short HEAD)"
hf download jesdga/Inkling-2-Mini-Basalt Inkling-2-Mini-NVFP4-q8.basalt --local-dir /root/b/official > $R/logs/dl_off.log 2>&1 &
cd src
/root/b/venv/bin/python tools/convert/convert.py --checkpoint $C/v2ck --out $C/Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt \
  --name "Inkling 2 Mini Unbound v2" --workers 24 > $R/logs/basalt_b.log 2>&1
log BASALT rc=$? "$(tail -n1 $R/logs/basalt_b.log | cut -c1-120)"
wait
sha256sum $C/Inkling-2-Mini-Unbound-v2-NVFP4-q8.basalt | tee -a $R/logs/status_ckb.txt
ls -l $C/*.basalt /root/b/official/*.basalt | tee -a $R/logs/status_ckb.txt
log CKB_DONE
