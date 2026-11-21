#!/usr/bin/env bash
# Bring up a fresh RTX 5090 pod as a frozen-engine SGLang eval worker (agent `fleet`). Run on the Mac:
#   bash ~/Code/inkling-abl/scripts/fleet/pod_bringup.sh <pod-name>
# Idempotent: rerun at any point. Pushes the scripts, harness and val data, launches pod_bringup_remote.sh under
# nohup (weights -> /srv/w/orig, pinned image, admin key, one frozen-engine server per GPU, probe check) and
# polls /root/logs/bringup.log until BRINGUP_DONE or BRINGUP_FAIL. Never starts or removes pods.
# Every pod-targeting lium call goes through scripts/ink_lium.sh (owner rule).
set -euo pipefail
export LIUM_NONINTERACTIVE=1
POD=${1:?usage: pod_bringup.sh <pod-name>}
R=/Users/nikshepsvn/Code/inkling-abl
H=$R/harness
ink() { bash "$R/scripts/ink_lium.sh" "$@"; }
ink exec "$POD" "mkdir -p /root/fleet /root/logs /srv/harness/cap /srv/data/val /srv/evals /srv/cands /srv/fleet" >/dev/null
for f in $R/scripts/pod_fetch.sh $R/scripts/parity/sglang_up.sh $R/scripts/fleet/sglang_up_gpu.sh \
         $R/scripts/fleet/pod_bringup_remote.sh $R/scripts/fleet/orig_hashes.txt $R/scripts/fleet/play_serve.sh; do
  ink scp "$POD" "$f" "/root/fleet/$(basename $f)" >/dev/null
done
for f in sgl_eval.py sgl_merge.py fleet_eval.py build_cand.py screen_ids.json borderline200_ids.json \
         refusal_phrases.txt refusal_phrases_seed.txt cap/mmlu300.jsonl cap/gsm8k100.jsonl; do
  ink scp "$POD" "$H/$f" "/srv/harness/$f" >/dev/null
done
for d in harmful borderline probes; do
  ink scp "$POD" "$R/data/val/$d.jsonl" "/srv/data/val/$d.jsonl" >/dev/null
done
echo "pushed files; starting remote bringup on $POD"
# Multi-GPU pods: GPU i serves on (lowest non-SSH port in the pod's Lium port map) + i, so each server is reachable.
BASE=$(ink ps --format json | python3 -c "
import json, sys
p = next(x for x in json.load(sys.stdin) if x['name'] == sys.argv[1])
print(min(int(k) for k in p['ports'] if k != '22') if p.get('gpu_count', 1) > 1 else 3001)" "$POD")
echo "base port $BASE"
PID=/root/fleet/bringup.pid
ink exec "$POD" "kill -0 \$(cat $PID 2>/dev/null) 2>/dev/null || { FLEET_PREP_ONLY=${FLEET_PREP_ONLY:-} FLEET_BASE_PORT=$BASE nohup bash /root/fleet/pod_bringup_remote.sh >/dev/null 2>&1 < /dev/null & echo \$! > $PID; }" >/dev/null
while true; do
  sleep 30
  last=$(ink exec "$POD" "tail -n 3 /root/logs/bringup.log" | grep -v '^Executing on' || true)
  echo "$(date -u +%H:%M:%SZ) $(echo "$last" | tail -n 1)"
  if echo "$last" | grep -q BRINGUP_DONE; then echo "$POD ready"; exit 0; fi
  if echo "$last" | grep -q BRINGUP_FAIL; then echo "$POD failed; see /root/logs/bringup.log"; exit 1; fi
done
