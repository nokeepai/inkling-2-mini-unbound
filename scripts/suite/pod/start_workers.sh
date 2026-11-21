#!/usr/bin/env bash
# Pod: record what each server serves at start, then one worker per GPU. Args: model per GPU (8 names).
cd /srv/suite
i=0
for m in "$@"; do
  [ -f /srv/suite/gpu$i.model ] || echo -n $m > /srv/suite/gpu$i.model
  if [ -f /srv/suite/worker_$i.pid ] && kill -0 $(cat /srv/suite/worker_$i.pid) 2>/dev/null; then echo "worker $i running"; else
    nohup python3 /srv/suite/suite_worker.py --gpu $i > /root/logs/worker_$i.log 2>&1 < /dev/null &
    echo $! > /srv/suite/worker_$i.pid
  fi
  i=$((i + 1))
done
sleep 5
ps aux | grep -c "[s]uite_worker.py"
