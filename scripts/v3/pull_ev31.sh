#!/bin/bash
# Mac: pull /root/v3/ev/<name> from ink-v31-eval, strip mass-casualty ids, Jev + Bison judge, aggregate. Counts only.
#   bash pull_ev.sh <name>
set -u
N=$1; R=~/Code/inkling-abl; P=~/.inkling-abl/v3_private; L="bash $R/scripts/ink_lium.sh"; D=$P/ev/$N
mkdir -p $D
$L exec ink-v31-eval "cd /root/v3/ev && tar czf /root/ev_$N.tgz $N" > /dev/null 2>&1
$L scp ink-v31-eval /root/ev_$N.tgz $P/ev/ev_$N.tgz --download > /dev/null 2>&1 || { echo "pull failed"; exit 1; }
tar xzf $P/ev/ev_$N.tgz -C $P/ev && rm -f $P/ev/ev_$N.tgz
# old sealed test: merge the shards into one file per mode
for m in off e09; do cat $D/old_s*/gens_harmful_$m.jsonl > $D/old_$m.jsonl 2>/dev/null; done
python3 $R/scripts/v3/mc_exclude.py strip $P/mc_ids.json $D/*.jsonl > /dev/null 2>&1
for f in test_off test_e09 rp_off old_off old_e09; do
  [ -s $D/$f.jsonl ] || continue
  python3 $R/scripts/v3/judge_v3.py $D/$f.jsonl $D/jev_$f.jsonl --backend jev --conc 16 > /dev/null 2>&1
  python3 $R/scripts/v3/judge_v3.py $D/$f.jsonl $D/bison_$f.jsonl --backend bison --conc 16 --max-usd 0.3 >> $D/bison.log 2>&1
  python3 $R/scripts/v3/agg_v3.py metrics --gens $D/$f.jsonl --jev $D/jev_$f.jsonl --bison $D/bison_$f.jsonl --name ${N}_$f --out $D/metrics_$f.json > /dev/null 2>&1
done
echo "PULLED $N $(date -u +%T)"; cat $D/kld.txt 2>/dev/null
