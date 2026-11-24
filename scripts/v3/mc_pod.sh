#!/bin/bash
# ON ink-v3-eval (after the R1 sampler PIDs were stopped): drop excluded ids from gens, restart on the filtered list.
set -u
python3 - <<'PY'
import json,glob,os
ids=set(json.load(open('/root/v3/mc_ids.json')))
for f in glob.glob('/root/v3/gens/*.jsonl'):
    rows=open(f).readlines(); keep=[l for l in rows if l.strip() and json.loads(l).get('id') not in ids]
    open(f,'w').writelines(keep); print(os.path.basename(f),'removed',len(rows)-len(keep))
PY
setsid nohup bash /root/v3/gen_r1_jobs.sh > /root/logs/gen_r1_jobs.log 2>&1 < /dev/null &
echo restarted
