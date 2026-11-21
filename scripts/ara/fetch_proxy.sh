#!/bin/bash
# Pull proxy.json for every c3 job from ink-research-1 into methods/ara/proxy/, then print the summary table.
R=~/Code/inkling-abl
L="bash $R/scripts/ink_lium.sh"
mkdir -p $R/methods/ara/proxy
$L exec ink-research-1 "cd /root/ara/jobs && for d in c3t*; do [ -f \$d/proxy.json ] && echo \$d \$(cat \$d/proxy.json); done" 2>/dev/null \
  | grep '^c3t' | while read cid js; do echo "$js" > $R/methods/ara/proxy/$cid.json; done
/private/tmp/claude-501/-Users-nikshepsvn-Code-innervale/140f1e4d-5898-4fb1-9611-0b847d426684/scratchpad/venv/bin/python $R/scripts/ara/summary.py
