#!/bin/bash
# ON ink-v3-eval, after FAST_DONE: big downloads for patching/merging/comparisons. No token needed (public repos).
set -u
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DOWNLOAD_TIMEOUT=60
( hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-BF16.gguf --local-dir /data/base > /data/logs/dl_bf16.log 2>&1; echo "DL_BF16 rc=$? $(date -u +%T)" ) &
( hf download nokeepai/Inkling-2-Mini-Unbound --revision v2.0 --local-dir /data/v2hf > /data/logs/dl_v2hf.log 2>&1; echo "DL_V2HF rc=$? $(date -u +%T)" ) &
( hf download nokeepai/Inkling-2-Mini-Unbound-GGUF Inkling-2-Mini-Unbound-Q8_0.gguf --revision v1.0 --local-dir /data/v10 > /data/logs/dl_v10.log 2>&1; echo "DL_V10 rc=$? $(date -u +%T)" ) &
( hf download AtomicChat/Inkling-2-Mini-Uncensored-GGUF Inkling-2-Mini-Uncensored-LoRA.gguf --local-dir /data/compare/models/atomic > /data/logs/dl_atomic.log 2>&1; echo "DL_ATOMIC rc=$? $(date -u +%T)" ) &
wait
echo "SLOW_DONE $(date -u +%FT%TZ)"
