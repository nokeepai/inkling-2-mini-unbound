#!/bin/bash
# Rent Phase 1 pods: a second PRO 6000 (HF baselines) and a second 5090 (judge). 2h TTL, budget caps.
export LIUM_NONINTERACTIVE=1
R=~/Code/inkling-abl
T=bb50bd64-e363-4adc-8111-1d44c8e36807
lium up cosmic-raven-30 -c 1 -t $T --name ink-research-2 --ttl 2h --budget 3.6 -y --json > $R/state/up_research2.json 2> $R/state/up_research2.err &
lium up gentle-hawk-b8 -c 1 -t $T --name ink-eval-2 --ttl 2h --budget 1.4 -y --json > $R/state/up_eval2.json 2> $R/state/up_eval2.err &
wait
for f in up_research2 up_eval2; do
  python3 -c "import json;d=json.load(open('$R/state/$f.json'))['pod'];print(d['name'],d['huid'],d['gpu_type'],d['price_per_hour'],d['removal_scheduled_at'])" 2>/dev/null || tail -5 $R/state/$f.err
done
for p in ink-research-2 ink-eval-2; do
  lium scp "$p" "$R/scripts/pod_fetch.sh" /root/pod_fetch.sh 2>&1 | tail -1
  lium exec "$p" "nohup bash /root/pod_fetch.sh > /root/fetch.log 2>&1 &" 2>&1 | tail -1
  lium exec "$p" "nohup docker pull lmsysorg/sglang:inkling-2-mini > /root/pull.log 2>&1 &" 2>&1 | tail -1
done
python3 $R/scripts/ledger.py
