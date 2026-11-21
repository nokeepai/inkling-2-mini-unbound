#!/bin/bash
# run_test.sh <name> <params.json local>: fit on the pod in the foreground and print the log
R=~/Code/inkling-abl
L="bash $R/scripts/ink_lium.sh"
$L exec ink-research-1 "mkdir -p /root/ara/jobs/$1" > /dev/null
$L scp ink-research-1 $2 /root/ara/jobs/$1/params.json > /dev/null
$L exec ink-research-1 "bash /root/ara/fit_job.sh $1; cat /root/logs/ara_$1.log"
