#!/bin/bash
# Owner priority (14:50Z): "OU done right" + attention+router stacks, plus near-zero-refusal candidates, to FULL at top priority.
R=/Users/nikshepsvn/Code/inkling-abl
E="python3 $R/scripts/fleet/enqueue.py"
cd "$R"
for c in methods/mpoa/cands/c1h_x4_oupro*; do $E "$(basename $c)" "$c" --stage full --priority 9.5 --by orchestrator --note "owner priority: OU done right"; done
for c in methods/experts/cands/c4cd6_c1t002 methods/experts/cands/c4cd3_c1t002 methods/experts/cands/c4rp8_c1t002; do $E "$(basename $c)" "$c" --stage full --priority 9.4 --by orchestrator --note "owner priority: attn+router stack"; done
for c in methods/ara/cands/c3x001 methods/ara/cands/c3t009 methods/ara/cands/c3x002; do [ -d "$c" ] && $E "$(basename $c)" "$c" --stage full --priority 9 --by orchestrator --note "near-zero screen refusal"; done
for c in methods/mpoa/cands/c1h_svd1_s2* methods/mpoa/cands/c1h_r1_svd1_* methods/mpoa/cands/c1h_r1_plain*; do $E "$(basename $c)" "$c" --stage full --priority 9 --by orchestrator --note "near-zero screen refusal"; done
