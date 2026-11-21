#!/bin/bash
# Stop my proxy daemon, run the preserve-set capture, push the new fit script, restart the proxy daemon.
R=~/Code/inkling-abl
L="bash $R/scripts/ink_lium.sh"
$L scp ink-research-1 $R/scripts/ara/ara_capture_pres.py /root/ara/ara_capture_pres.py > /dev/null
$L scp ink-research-1 $R/scripts/ara/ara_fit.py /root/ara/ara_fit.py > /dev/null
$L exec ink-research-1 "kill 28659; sleep 5; cd /root/ara && python ara_capture_pres.py > /root/logs/ara_pcap.log 2>&1; tail -2 /root/logs/ara_pcap.log; nohup python ara_proxy.py > /root/logs/ara_proxy.log 2>&1 & echo proxy_restarted"
