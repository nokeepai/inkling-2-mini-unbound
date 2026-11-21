#!/bin/bash
# research-3: after chain2's three runs, the layer-limited adapter variants.
cd /root/exp
until [ "$(grep -c HOOKS_DONE /root/logs/exp_chain2b.log)" -ge 3 ]; do sleep 10; done
ROWS_BS=120 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python hooks_rows.py moepostL attn+moepostL topexp8 attn+topexp8
