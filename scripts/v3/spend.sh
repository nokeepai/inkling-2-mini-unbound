#!/bin/bash
# Mac: append a spend snapshot (Lium pods named ink-v3-* from `lium ps`, plus API balances) to v3/spend.log.
# Why: the cap is on measured spend, not my estimate; this runs every 10 min in the background.
set -u
R=~/Code/inkling-abl
while true; do
  T=$(date -u +%FT%TZ)
  PODS=$(lium ps 2>/dev/null | grep -E 'ink-v3' | awk '{for(i=1;i<=NF;i++) if($i ~ /^\$[0-9.]+(\/\$[0-9.]+)?$/) printf "%s ", $i; print $NF}')
  API=$(cd ~/Code/uncensoredbench && python3 - <<'EOF' 2>/dev/null
import sys, json, urllib.request
sys.path.insert(0, "tools"); import pilot
def get(u, k):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"Authorization": "Bearer " + k}), timeout=30))
o = get("https://openrouter.ai/api/v1/credits", pilot.key("openrouter"))["data"]
v = get("https://api.venice.ai/api/v1/api_keys/rate_limits", pilot.key("venice"))
v = v.get("data", v).get("balances", {}).get("USD")
print(f"or_left={o['total_credits'] - o['total_usage']:.2f} venice={v:.2f}")
EOF
)
  echo "$T | $API | $(echo $PODS | tr '\n' ' ')" >> $R/v3/spend.log
  [ "${1:-}" = once ] && break
  sleep 600
done
