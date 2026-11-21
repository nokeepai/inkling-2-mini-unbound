#!/bin/bash
# Push ara pod-side scripts and the harness builder to ink-research-1.
R=~/Code/inkling-abl
L="bash $R/scripts/ink_lium.sh"
for f in ara_fit.py check_conv.py fit_job.sh verify_build.py; do
  $L scp ink-research-1 $R/scripts/ara/$f /root/ara/$f > /dev/null
done
$L scp ink-research-1 $R/harness/build_cand.py /root/ara/build_cand.py > /dev/null
echo pushed
