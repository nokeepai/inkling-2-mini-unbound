#!/bin/bash
# research-3 was recreated (lunar-wolf-db): install pinned TF, loader (pointed at /root/w/orig2), diaglib.
R=~/Code/inkling-abl
S="bash $R/scripts/ink_lium.sh"
P=ink-research-3
$S exec $P "mkdir -p /root/parity /root/diag /root/logs" >/dev/null
$S scp $P $R/scripts/parity/pod_install_hf.sh /root/pod_install_hf.sh >/dev/null
$S scp $P $R/scripts/parity/hf_load.py /root/parity/hf_load.py >/dev/null
$S scp $P $R/scripts/diag/diaglib.py /root/diag/diaglib.py >/dev/null
$S exec $P "sed -i 's#^ORIG = \"/root/w/orig\"#ORIG = \"/root/w/orig2\"#' /root/parity/hf_load.py; grep -n '^ORIG' /root/parity/hf_load.py; nohup bash /root/pod_install_hf.sh > /root/logs/install_exp.log 2>&1 & sleep 1; tail -c 300 /root/fetch.log; du -sh /root/w/*; nvidia-smi --query-gpu=memory.used --format=csv,noheader; python3 -c 'import torch;print(torch.__version__)'"
