#!/bin/bash
# ON ink-v21-rel: load each release GGUF in llama-server and answer 6 benign prompts with thinking off and on (greedy).
# Logs counts only: non-empty answers, think-cap hits, and exact-answer checks (no generated text is printed).
#   bash smoke.sh <gguf_dir>
set -u
export V3_REFUSALS=/root/harness/refusal_phrases.txt CMP_TOK=/data/compare/models/tok
G=$1; O=/data/rel/smoke; mkdir -p $O; cd /root/v11b
python3 - <<'PY'
import json
rows = [("s1", "What is 17 * 23? Reply with just the number.", "391"),
        ("s2", "What is the capital of Australia? One word.", "canberra"),
        ("s3", "Write a Python function that reverses a string.", "def "),
        ("s4", "Name the chemical symbol for gold. Just the symbol.", "au"),
        ("s5", "Write a haiku about autumn rain.", ""),
        ("s6", "If a train travels 60 km in 45 minutes, what is its speed in km/h? Just the number.", "80")]
open("/data/rel/smoke/prompts.jsonl", "w").writelines(json.dumps({"id": i, "text": t, "category": "smoke", "expect": e}) + "\n" for i, t, e in rows)
PY
for f in $G/Inkling-2-Mini-Unbound-v2.1-*.gguf; do
  T=$(basename $f .gguf | sed 's/.*v2.1-//'); [ "$T" = BF16 ] && continue
  bash servers.sh $f smoke_$T 0 4 16384 > $O/srv_$T.txt 2>&1
  for e in 0 0.9; do
    python3 gen_v3.py --ports 8100 --inp $O/prompts.jsonl --out $O/${T}_e$e.jsonl --effort $e --greedy --think-cap 2048 --ans-cap 1024 --conc 4 \
      > $O/log_${T}_e$e.txt 2>&1
  done
  kill $(cat /root/logs/srv_0.pid) 2>/dev/null; sleep 5
  python3 - "$T" <<'PY'
import json, sys
T = sys.argv[1]
exp = {json.loads(l)["id"]: json.loads(l)["expect"] for l in open("/data/rel/smoke/prompts.jsonl")}
for e in ("0", "0.9"):
    rows = [json.loads(l) for l in open(f"/data/rel/smoke/{T}_e{e}.jsonl")]
    ans = lambda r: (r.get("response") or r.get("answer") or "").lower()
    ok = sum(1 for r in rows if exp[r["id"]] and exp[r["id"]] in ans(r))
    print(f"SMOKE {T} effort={e} rows={len(rows)} nonempty={sum(1 for r in rows if ans(r).strip())} "
          f"capped={sum(1 for r in rows if r.get('cap_hit'))} exact={ok}/{sum(1 for v in exp.values() if v)}", flush=True)
PY
done
echo SMOKE_DONE
