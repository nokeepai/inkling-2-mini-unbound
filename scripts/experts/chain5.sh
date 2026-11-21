#!/bin/bash
# research-3: after chain4, the lead's router-row projection idea (rprojK: span{h_l, Q} out of top-K pairs' rows).
cd /root/exp
until grep -qE 'HOOKS_DONE|Traceback' /root/logs/exp_chain4.log; do sleep 10; done
export ROWS_BS=60 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python hooks_rows2.py attn+rproj8 attn+rproj32 rproj32
