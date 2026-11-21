#!/bin/bash
# Rent Phase 4 capability-suite pods: 8x5090 (France) + 1 PRO 6000 (US, long context/vision). 2h TTL (ttl_guard extends).
export LIUM_NONINTERACTIVE=1
R=~/Code/inkling-abl
T=bb50bd64-e363-4adc-8111-1d44c8e36807
bash $R/scripts/ink_lium.sh up 1f4e9f40-dc3a-444c-b475-9ed75ba72528 -c 8 -t $T --name ink-suite-1 --ttl 2h --budget 16 --verify-gpus -y --json > $R/state/up_suite1.json 2> $R/state/up_suite1.err &
bash $R/scripts/ink_lium.sh up 375767d4-e378-4f03-b3cc-6d6ed32caf31 -c 1 -t $T --name ink-suite-lc --ttl 2h --budget 5.2 -y --json > $R/state/up_suitelc.json 2> $R/state/up_suitelc.err &
wait
tail -1 $R/state/up_suite1.err; tail -1 $R/state/up_suitelc.err
for p in ink-suite-1 ink-suite-lc; do
  bash $R/scripts/ink_lium.sh scp $p "$R/scripts/pod_fetch.sh" /root/pod_fetch.sh 2>&1 | tail -1
  bash $R/scripts/ink_lium.sh exec $p "nohup bash /root/pod_fetch.sh > /root/fetch.log 2>&1 &" 2>&1 | tail -1
  bash $R/scripts/ink_lium.sh exec $p "nohup docker pull lmsysorg/sglang@sha256:421aed3ac11741381eac350bf7b1db28a0cafb6ae49b57ae78fba86a16a32306 > /root/pull.log 2>&1 &" 2>&1 | tail -1
done
sleep 75
for p in ink-suite-1 ink-suite-lc; do bash $R/scripts/ink_lium.sh exec $p "du -sh /root/w 2>/dev/null" 2>&1 | tail -1; done
python3 $R/scripts/ledger.py | sed -n 3,6p
