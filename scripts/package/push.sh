#!/bin/bash
# Push packaging scripts, build_cand and a candidate dir to a pod under /root/pkg.
set -e
P=${1:?pod}; CAND=${2:?cand_dir}
R=~/Code/inkling-abl; L="bash $R/scripts/ink_lium.sh"
cp $R/harness/build_cand.py $R/scripts/package/build_cand.py.copy
$L exec "$P" "mkdir -p /root/pkg/scripts /root/pkg/cands /root/pkg/out /root/logs"
$L rsync "$P" $R/scripts/package/ /root/pkg/scripts/ --exclude '*.copy'
$L scp "$P" $R/scripts/package/build_cand.py.copy /root/pkg/scripts/build_cand.py
$L rsync "$P" "$CAND" /root/pkg/cands/$(basename $CAND)
$L exec "$P" "ls -la /root/pkg/scripts /root/pkg/cands/*"
