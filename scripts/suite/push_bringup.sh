#!/usr/bin/env bash
# Mac: push suite scripts + ou candidate to a pod and start bringup_remote.sh under nohup.
set -euo pipefail
POD=${1:?pod}
R=/Users/nikshepsvn/Code/inkling-abl
ink() { bash "$R/scripts/ink_lium.sh" "$@"; }
ink exec "$POD" "mkdir -p /root/suite /root/logs /srv/suite /srv/cands/ou /srv/w" >/dev/null
ink scp "$POD" $R/scripts/fleet/orig_hashes.txt /root/suite/orig_hashes.txt >/dev/null
for f in suite_serve.sh bringup_remote.sh; do ink scp "$POD" $R/scripts/suite/$f /srv/suite/$f >/dev/null; done
ink scp "$POD" $R/harness/sgl_merge.py /srv/suite/sgl_merge.py >/dev/null
for f in spec.json sha256.txt tensors.safetensors; do ink scp "$POD" $R/methods/baselines/cands/ou/$f /srv/cands/ou/$f >/dev/null; done
ink exec "$POD" "nohup bash /srv/suite/bringup_remote.sh > /dev/null 2>&1 < /dev/null &" >/dev/null
echo "started bringup on $POD"
