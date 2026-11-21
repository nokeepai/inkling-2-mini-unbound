#!/bin/bash
set -e
R=~/Code/inkling-abl
L="bash $R/scripts/ink_lium.sh"
$L exec ink-research-1 "mkdir -p /root/ara/data /root/ara/cap /root/logs"
$L scp ink-research-1 $R/data/train/harmful.jsonl /root/ara/data/harmful.jsonl
$L scp ink-research-1 $R/data/train/harmless.jsonl /root/ara/data/harmless.jsonl
$L scp ink-research-1 $R/scripts/ara/ara_capture.py /root/ara/ara_capture.py
$L exec ink-research-1 "cd /root/ara && nohup python ara_capture.py 256 > /root/logs/ara_cap.log 2>&1 &"
