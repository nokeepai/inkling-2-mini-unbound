#!/usr/bin/env bash
# Pod: start one frozen-engine server per GPU. Args: model dir per GPU, e.g. /w/orig /w/orig ... (8 args). Port 4000+i.
mkdir -p /root/logs
i=0
for m in "$@"; do
  bash /srv/suite/suite_serve.sh $i $((4000 + i)) $m > /root/logs/serve_$i.log 2>&1 &
  i=$((i + 1))
done
wait
for j in $(seq 0 $((i - 1))); do echo "gpu$j: $(tail -n 1 /root/logs/serve_$j.log)"; done
