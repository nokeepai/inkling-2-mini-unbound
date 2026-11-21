#!/usr/bin/env bash
# Mac: push pod scripts, prep data, show server state, run smoke.
set -uo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
ink() { bash "$R/scripts/ink_lium.sh" "$@"; }
bash $R/scripts/suite/push_pod.sh
ink exec ink-suite-1 "cat /root/logs/start_servers.log; cd /srv/suite && python3 prep_data.py 2>&1 | tail -8; python3 -c 'from evalplus.data import get_human_eval_plus; print(len(get_human_eval_plus()))' 2>&1 | tail -1"
ink exec ink-suite-1 "bash /srv/suite/smoke.sh > /root/logs/smoke.log 2>&1; cat /root/logs/smoke.log"
