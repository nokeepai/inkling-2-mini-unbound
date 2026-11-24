#!/bin/bash
# Mac: loop: pull gens/eval files from the pods and Jev-judge them incrementally (judge_v3 is resumable).
#   bash pull_judge.sh [once]
# Training candidates -> ~/.inkling-abl/v3_private/r1 ; v2 baseline evals -> ~/.inkling-abl/v3_private/ev/v2
set -u
R=~/Code/inkling-abl; P=~/.inkling-abl/v3_private; L="bash $R/scripts/ink_lium.sh"
mkdir -p $P/r1 $P/ev/v2
pull() {  # pod remote_dir local_dir
  for f in $($L exec $1 "ls $2/*.jsonl 2>/dev/null" 2>/dev/null | grep '\.jsonl$'); do
    $L scp $1 "$f" "$3/$(basename $f).part" --download > /dev/null 2>&1 || continue
    # the pod may be mid-append: keep only complete JSON lines
    python3 -c "
import json,sys
src,dst=sys.argv[1],sys.argv[2]; ok=[]
for l in open(src):
    try: json.loads(l); ok.append(l)
    except Exception: pass
open(dst,'w').writelines(ok)" "$3/$(basename $f).part" "$3/$(basename $f)" && rm -f "$3/$(basename $f).part"
  done
}
while true; do
  pull ink-v3-eval /root/v3/gens $P/r1
  pull ink-v3-gen2 /root/v3/gens $P/r1
  pull ink-v3-gen2 /root/v3/eval_v2 $P/ev/v2
  # owner rule: mass-casualty-category ids never enter v3 data or scoring
  python3 $R/scripts/v3/mc_exclude.py strip $P/mc_ids.json $P/r1/*.jsonl $P/ev/v2/*.jsonl > /dev/null 2>&1
  for f in $P/r1/*.jsonl $P/ev/v2/*.jsonl; do
    b=$(basename $f); case $b in jev_*|bison_*) continue;; esac
    python3 $R/scripts/v3/judge_v3.py $f $(dirname $f)/jev_$b --backend jev --conc 16 > /dev/null 2>&1
  done
  echo "$(date -u +%T) cycle done"
  [ "${1:-}" = once ] && break
  sleep 120
done
