#!/bin/bash
# Rent replacement fleet pod (8x5090, HK), start fetch + image pull, then measure download speed after 90 s.
export LIUM_NONINTERACTIVE=1
R=~/Code/inkling-abl
bash $R/scripts/ink_lium.sh up dc48061c-8e25-48ec-b11d-f45e313d096d -c 8 -t bb50bd64-e363-4adc-8111-1d44c8e36807 --name ink-fleet-2 --ttl 2h --budget 11.5 --verify-gpus -y --json > $R/state/up_fleet2.json 2> $R/state/up_fleet2.err
tail -2 $R/state/up_fleet2.err
bash $R/scripts/ink_lium.sh scp ink-fleet-2 "$R/scripts/pod_fetch.sh" /root/pod_fetch.sh 2>&1 | tail -1
bash $R/scripts/ink_lium.sh exec ink-fleet-2 "nohup bash /root/pod_fetch.sh > /root/fetch.log 2>&1 &" 2>&1 | tail -1
bash $R/scripts/ink_lium.sh exec ink-fleet-2 "nohup docker pull lmsysorg/sglang:inkling-2-mini > /root/pull.log 2>&1 &" 2>&1 | tail -1
sleep 90
bash $R/scripts/ink_lium.sh exec ink-fleet-2 "du -sh /root/w 2>/dev/null; tail -c 300 /root/fetch.log; nvidia-smi -L | wc -l" 2>&1 | tail -6
