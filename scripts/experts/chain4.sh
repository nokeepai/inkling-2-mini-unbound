#!/bin/bash
# research-3: one row-masked run, 60-row chunks (one condition per chunk), trimmed condition list.
cd /root/exp
export ROWS_BS=60 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python hooks_rows.py base attn moepost attn+moepost moepostL attn+moepostL attn+mask32 attn+cond32_d6 attn+topexp8 mask32
