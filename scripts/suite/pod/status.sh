#!/usr/bin/env bash
# Pod: one-screen status of the suite queue.
cd /srv/suite
date -u +%H:%M:%SZ
python3 - <<'EOF'
import json, sys
sys.path.insert(0, "/srv/suite")
import suite_worker as W
cl = [json.loads(l) for l in open(W.C)]
st = {}
for c in cl:
    st[c["key"]] = c
q = W.queue()
by = {}
for r in q:
    s = st.get(W.key(r), {}).get("state", "todo")
    by.setdefault(s, []).append(W.key(r))
for s, ks in by.items():
    print(s, len(ks), " ".join(ks) if s in ("claimed", "error") else "")
EOF
for i in 0 1 2 3 4 5 6 7; do echo "g$i [$(cat gpu$i.model)] $(tail -n 1 /root/logs/worker_$i.events)"; done
ls out/*/*/score.json 2>/dev/null | grep -v smoke
