#!/bin/bash
# Move C4 state to ink-research-3: scripts, data, ranking from research-1; loader pointed at /root/w/orig2.
R=~/Code/inkling-abl
S="bash $R/scripts/ink_lium.sh"
T=$R/methods/experts/pod
mkdir -p $T
$S scp ink-research-1 /root/exp/out/ranking.json $T/ranking.json --download >/dev/null
$S scp ink-research-1 /root/exp/out/diag_math.json $R/methods/experts/diag_math.json --download >/dev/null
POD=ink-research-3 bash $R/scripts/experts/upload.sh data >/dev/null
$S scp ink-research-3 $R/scripts/experts/hooks_rows.py /root/exp/hooks_rows.py >/dev/null
$S scp ink-research-3 $T/ranking.json /root/exp/out/ranking.json >/dev/null
$S exec ink-research-3 "sed -i 's#^ORIG = \"/root/w/orig\"#ORIG = \"/root/w/orig2\"#' /root/parity/hf_load.py; grep -n '^ORIG' /root/parity/hf_load.py; ls /root/exp /root/exp/data /root/exp/out; du -sh /root/w/orig2"
