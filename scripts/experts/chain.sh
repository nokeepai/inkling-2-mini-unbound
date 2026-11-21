#!/bin/bash
# Pod-side: wait for weights + install, then localise -> analyse -> hook conditions.
cd /root/exp
[ "${NOWAIT:-0}" = 1 ] || until grep -q FETCH_DONE /root/fetch.log 2>/dev/null && grep -q INSTALL_DONE /root/logs/install.log 2>/dev/null; do sleep 20; done
echo "$(date -u +%T) ready"
[ -f /root/exp/out/ranking.json ] || { python localise.py && python analyse_loc.py; } || exit 1
python hooks_eval.py ${CONDS:-base moepost attn attn+moepost mask8 mask32 attn+mask32 cond32_d6 attn+cond32_d6 moepreS topexp32 attn+topexp32}
