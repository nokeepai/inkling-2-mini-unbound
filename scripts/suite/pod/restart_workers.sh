#!/usr/bin/env bash
# Pod: stop suite workers (their running bench children keep going and are adopted), fix the admin key, requeue
# unfinished jobs, start workers again.
cd /srv/suite
for i in 0 1 2 3 4 5 6 7; do [ -f worker_$i.pid ] && kill $(cat worker_$i.pid) 2>/dev/null; done
sleep 2
ps aux | grep -c "[s]uite_worker.py"
# one clean key (the 8 parallel first launches raced and wrote a two-line key)
head -n 1 /srv/admin_key | tr -d '\n' > /srv/admin_key.tmp && mv /srv/admin_key.tmp /srv/admin_key
python3 requeue.py
bash start_workers.sh orig orig orig orig ou ou ou ou
