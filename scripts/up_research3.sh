#!/bin/bash
# Rent ink-research-3 (PRO 6000) for the C4 experts feasibility study; 2h TTL, budget cap.
export LIUM_NONINTERACTIVE=1
R=~/Code/inkling-abl
bash $R/scripts/ink_lium.sh up c95c1848-93d3-4f82-864d-746319eee65f -c 1 -t bb50bd64-e363-4adc-8111-1d44c8e36807 --name ink-research-3 --ttl 2h --budget 3.4 -y --json > $R/state/up_research3.json 2> $R/state/up_research3.err
tail -2 $R/state/up_research3.err
bash $R/scripts/ink_lium.sh scp ink-research-3 "$R/scripts/pod_fetch.sh" /root/pod_fetch.sh 2>&1 | tail -1
bash $R/scripts/ink_lium.sh exec ink-research-3 "nohup bash /root/pod_fetch.sh > /root/fetch.log 2>&1 &" 2>&1 | tail -1
python3 $R/scripts/ledger.py | sed -n 3,6p
