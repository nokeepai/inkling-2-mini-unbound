#!/bin/bash
# ON ink-v21-serve (2x RTX PRO 6000): build the v2.1 HF checkpoint (same merge + splice as the release build), then serve
# it with vLLM (data-parallel 2, OpenAI API, API key) behind a Cloudflare quick tunnel. Progress: /root/logs/serve.log
set -u
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DOWNLOAD_TIMEOUT=60
L=/root/logs/serve.log; mkdir -p /root/logs /root/serve /data/srv; log() { echo "$(date -u +%T) $*" >> $L; }
IMG=${IMG:-vllm/vllm-openai:inkling-2-mini}
log START
( docker pull -q $IMG > /root/logs/pull.log 2>&1 || { IMG=vllm/vllm-openai:latest; docker pull -q $IMG >> /root/logs/pull.log 2>&1; }
  echo $IMG > /root/serve/img; log PULL "$(cat /root/serve/img)" ) &
( curl -sL -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    && chmod +x /usr/local/bin/cloudflared; log CLOUDFLARED rc=$? ) &
pip install -q safetensors hf_transfer "huggingface_hub[cli]" > /dev/null 2>&1
mkdir -p /root/stage && tar xzf /root/stage31.tgz -C /root/stage
hf download nokeepai/Inkling-2-Mini-Unbound chat_template.jinja config.json generation_config.json hf_quant_config.json \
  model-00001-of-00001.safetensors model.safetensors.index.json mtp.safetensors processor_config.json \
  special_tokens_map.json tiktoken/tokenizer.model tokenizer.json tokenizer_config.json \
  --revision v2.0 --local-dir /data/srv/v2hf > /root/logs/dl_v2hf.log 2>&1; log DL_V2HF rc=$?
cd /root/stage/v11b
python3 merge_stack.py /data/srv/v2hf hf_name_map.json /data/srv/v21_tensors.safetensors \
  /root/stage/v3/loras/sft1.pt:1.0 /root/stage/v3/loras/dpo2.pt:1.0 > /root/logs/merge.log 2>&1; log MERGE rc=$? "$(tail -n1 /root/logs/merge.log)"
python3 splice_ckpt.py /data/srv/v2hf /data/srv/v21_tensors.safetensors /data/srv/v21ck > /root/logs/splice.log 2>&1
log SPLICE rc=$? "$(tail -n1 /root/logs/splice.log)"
sha256sum /data/srv/v21_tensors.safetensors /data/srv/v21ck/model-00001-of-00001.safetensors >> $L
wait
[ -f /root/serve/key ] || { python3 -c 'import secrets;print("sk-inkling-"+secrets.token_urlsafe(24))' > /root/serve/key; chmod 600 /root/serve/key; }
bash /root/v21/serve_run.sh
