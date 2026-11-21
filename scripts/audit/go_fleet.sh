#!/bin/bash
# Push tooling + finalists + sealed test to ink-fleet-2 and start the core sealed jobs on all 8 GPUs.
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
L="bash $R/scripts/ink_lium.sh"
P=$R/audit/.sealed_plain/sealed/test
bash $R/scripts/audit/push.sh ink-fleet-2 $R/methods/baselines/cands/ou $R/methods/mpoa/cands/c1h_r1_svd1_s3_L4 $R/methods/mpoa/cands/c1dt001 | tail -3
$L exec ink-fleet-2 "mkdir -p /srv/audit/data/test && chmod 700 /srv/audit/data /srv/audit/data/test" >/dev/null
$L rsync ink-fleet-2 "$P/" /srv/audit/data/test/ >/dev/null
$L exec ink-fleet-2 "chmod 600 /srv/audit/data/test/*; wc -l /srv/audit/data/test/*.jsonl | tail -1; cd /srv/audit/scripts && setsid nohup bash run_jobs.sh /srv/audit/data test /srv/audit/robust_test 4096 jobs_test_core.txt > /root/logs/audit_jobs_core.log 2>&1 < /dev/null & echo CORE_STARTED"
