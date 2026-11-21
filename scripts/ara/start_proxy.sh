#!/bin/bash
# Push and start the ara proxy daemon on ink-research-1.
R=~/Code/inkling-abl
L="bash $R/scripts/ink_lium.sh"
$L scp ink-research-1 $R/scripts/ara/ara_proxy.py /root/ara/ara_proxy.py > /dev/null
$L exec ink-research-1 "cd /root/ara && nohup python ara_proxy.py > /root/logs/ara_proxy.log 2>&1 & echo started"
