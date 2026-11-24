#!/bin/bash
# ON ink-v3-train: pinned transformers PR, v2 HF checkpoint (public nokeepai/Inkling-2-Mini-Unbound @ v2.0) as the base.
set -u
mkdir -p /root/logs /root/w /root/parity /root/v3/runs
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DOWNLOAD_TIMEOUT=60
echo "START $(date -u +%FT%TZ)"
pip install -q hf_transfer "huggingface_hub[cli]" 2>&1 | tail -1
( hf download nokeepai/Inkling-2-Mini-Unbound --revision v2.0 --local-dir /root/w/v2 > /root/logs/dl_v2.log 2>&1; echo "DL_V2 rc=$? $(date -u +%T)" ) &
pip install -q "git+https://github.com/huggingface/transformers@a10ebc3c1c44b01d0ee825d48b8d0de3253e4539" \
  accelerate==1.16.0 safetensors==0.8.0 sentencepiece==0.2.3 protobuf==7.36.2 tokenizers==0.23.3 2>&1 | tail -2
python3 -c "import transformers,torch;print('TF', transformers.__version__, torch.__version__)"
wait
du -sh /root/w/v2; ls /root/w/v2
echo "SETUP_DONE $(date -u +%FT%TZ)"
