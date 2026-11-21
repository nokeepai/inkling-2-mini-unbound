#!/bin/bash
# Push finalists + sealed test to ink-research-1 and start the test KL run (orig gen 1000 -> cache -> ou, U, F).
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
L="bash $R/scripts/ink_lium.sh"
P=$R/audit/.sealed_plain/sealed/test
bash $R/scripts/audit/push.sh ink-research-1 $R/methods/baselines/cands/ou $R/methods/mpoa/cands/c1h_r1_svd1_s3_L4 $R/methods/mpoa/cands/c1dt001 | tail -3
$L exec ink-research-1 "mkdir -p /srv/audit/data/test && chmod 700 /srv/audit/data /srv/audit/data/test" >/dev/null
$L rsync ink-research-1 "$P/" /srv/audit/data/test/ >/dev/null
$L exec ink-research-1 "chmod 600 /srv/audit/data/test/*; wc -l /srv/audit/data/test/*.jsonl | tail -1; cd /srv/audit/scripts && setsid nohup bash kl_run.sh /srv/audit/data/test/harmless.jsonl test 1000 /srv/audit/cands/ou /srv/audit/cands/c1h_r1_svd1_s3_L4 /srv/audit/cands/c1dt001 > /root/logs/audit_kl_test.log 2>&1 < /dev/null & echo KL_STARTED"
