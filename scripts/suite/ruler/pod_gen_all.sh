#!/bin/bash
# Pod-side: generate all 13 RULER tasks in parallel processes (each seeds itself). Logs: /root/logs/ruler_data_<task>.log
N32=${1:-25}; N128=${2:-10}
cd /srv/suite
for t in niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2; do
  python3 /srv/suite/ruler/gen_ruler_data.py $t $N32 $N128 > /root/logs/ruler_data_$t.log 2>&1 &
done
wait
echo ALL_DATA_DONE
ls -la /srv/suite/data/ruler
du -sh /srv/suite/data/ruler
