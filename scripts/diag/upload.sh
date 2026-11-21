#!/bin/bash
export LIUM_NONINTERACTIVE=1
R=~/Code/inkling-abl
P=ink-research-1
lium exec $P "mkdir -p /root/diag/data /root/logs" >/dev/null
for f in diaglib.py leak.py capture.py attrib.py atoms_prompts.json; do
  [ -f $R/scripts/diag/$f ] && lium scp $P $R/scripts/diag/$f /root/diag/$f >/dev/null
done
if [ "$1" = "data" ]; then
  lium scp $P $R/vendor/ckpt/ou/edit.pt /root/diag/edit.pt
  lium scp $P $R/data/train/harmful.jsonl /root/diag/data/harmful.jsonl
  lium scp $P $R/data/train/harmless.jsonl /root/diag/data/harmless.jsonl
  lium scp $P $R/harness/refusal_phrases_seed.txt /root/diag/data/refusal_phrases_seed.txt
  python3 -c "
import json
qs=[json.loads(l)['question'] for l in open('$R/harness/cap/gsm8k100.jsonl')]
json.dump(qs,open('/tmp/gsm_q.json','w'))"
  lium scp $P /tmp/gsm_q.json /root/diag/data/gsm_q.json
fi
lium exec $P "ls -la /root/diag /root/diag/data"
