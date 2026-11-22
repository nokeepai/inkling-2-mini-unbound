#!/bin/bash
# ON ink-v11-res: pinned transformers PR, orig ckpt @ ed40e7da, v1.0 = orig + out/final/edit.pt (bit-exact apply).
set -u
mkdir -p /root/logs /root/w /root/parity /root/v11b
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "START $(date -u +%FT%TZ)"
pip install -q hf_transfer "huggingface_hub[cli]" 2>&1 | tail -1
( hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da --local-dir /root/w/orig > /root/logs/dl_orig.log 2>&1; echo "DL_ORIG_DONE rc=$?" >> /root/logs/dl_orig.log ) &
pip install -q "git+https://github.com/huggingface/transformers@a10ebc3c1c44b01d0ee825d48b8d0de3253e4539" \
  accelerate==1.16.0 safetensors==0.8.0 sentencepiece==0.2.3 protobuf==7.36.2 tokenizers==0.23.3 2>&1 | tail -2
python3 -c "import transformers,torch;print('TF', transformers.__version__, torch.__version__)"
wait
tail -1 /root/logs/dl_orig.log
python3 /root/v11b/apply_edit.py /root/v11b/edit.pt /root/w/orig /root/w/v10 && echo APPLY_OK
du -sh /root/w/*
echo "SETUP_DONE $(date -u +%FT%TZ)"
