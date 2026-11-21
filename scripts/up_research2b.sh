#!/bin/bash
# Replace expired ink-research-2 with a US PRO 6000 (KL worker + judge). 2h TTL, budget cap.
export LIUM_NONINTERACTIVE=1
R=~/Code/inkling-abl
bash $R/scripts/ink_lium.sh up 8238f069-4cf9-4165-bd44-017541e1017b -c 1 -t bb50bd64-e363-4adc-8111-1d44c8e36807 --name ink-research-2b --ttl 2h --budget 3.6 -y --json > $R/state/up_research2b.json 2> $R/state/up_research2b.err
tail -2 $R/state/up_research2b.err
bash $R/scripts/ink_lium.sh scp ink-research-2b "$R/scripts/pod_fetch.sh" /root/pod_fetch.sh 2>&1 | tail -1
bash $R/scripts/ink_lium.sh exec ink-research-2b "nohup bash /root/pod_fetch.sh > /root/fetch.log 2>&1 &" 2>&1 | tail -1
python3 $R/scripts/ledger.py | sed -n 3,6p
