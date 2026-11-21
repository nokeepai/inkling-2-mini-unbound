#!/usr/bin/env bash
# Copy judge.py + smoke inputs to ink-eval-2 and run the CLI on CPU only.
set -euo pipefail
export LIUM_NONINTERACTIVE=1
POD=ink-eval-2
lium exec "$POD" "mkdir -p /srv/judge/harness /root/logs"
lium scp "$POD" /Users/nikshepsvn/Code/inkling-abl/harness/judge.py /srv/judge/harness/judge.py
lium scp "$POD" /Users/nikshepsvn/Code/inkling-abl/scripts/judge/smoke_in.jsonl /srv/judge/smoke_in.jsonl
lium exec "$POD" "CUDA_VISIBLE_DEVICES='' python3 /srv/judge/harness/judge.py /srv/judge/smoke_in.jsonl /srv/judge/smoke_out.jsonl --batch-size ${BS:-4} 2>&1 | grep -v -i warn | tail -n 5"
lium exec "$POD" "python3 -c \"import json;[print(r['id'],r['expect'],'refused',r['refused'],'score',r['judge_score'],'p',r['probs'],r['raw'][:40]) for r in map(json.loads,open('/srv/judge/smoke_out.jsonl'))]\""
