#!/bin/bash
# ON pod: quick health/speed benchmark (network down/up, disk, GPU). Prints one line per test.
set -u
export HF_HUB_ENABLE_HF_TRANSFER=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c
df -h /root /data 2>/dev/null | tail -n +2
nproc; free -g | sed -n 2p
dd if=/dev/zero of=/data/_ddtest bs=1M count=4000 oflag=direct 2>&1 | tail -1; dd if=/data/_ddtest of=/dev/null bs=1M iflag=direct 2>&1 | tail -1; rm -f /data/_ddtest
pip install -q hf_transfer "huggingface_hub[cli]" >/dev/null 2>&1
t=$(date +%s); hf download ggml-org/Inkling-2-Mini-GGUF mmproj-Inkling-2-Mini-BF16.gguf --local-dir /data/_bench >/dev/null 2>&1; e=$(( $(date +%s)-t )); s=$(du -sm /data/_bench | cut -f1); echo "HF_DL ${s}MB in ${e}s = $(( s / (e>0?e:1) )) MB/s"
t=$(date +%s); curl -4 -s -o /dev/null -w "%{speed_upload}\n" -X POST --data-binary @/data/_bench/mmproj-Inkling-2-Mini-BF16.gguf https://httpbin.org/post --max-time 60 | awk '{printf "UP %.1f MB/s\n",$1/1e6}'
rm -rf /data/_bench
echo BENCH_DONE
