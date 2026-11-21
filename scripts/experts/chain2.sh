#!/bin/bash
# research-3: three row-masked hook runs of 4 conditions each (partial results land per run).
cd /root/exp
until grep -q INSTALL_DONE /root/logs/install_exp.log 2>/dev/null; do sleep 10; done
echo "$(date -u +%T) ready"
python hooks_rows.py base moepost attn attn+moepost || exit 1
python hooks_rows.py mask8 mask32 attn+mask32 attn+cond32_d6 || exit 1
python hooks_rows.py cond32_d6 moepreS topexp32 attn+topexp32
