#!/usr/bin/env bash
# Mac: push finalists from suite/finalists.json to ink-suite-1 and build+merge them there.
set -uo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
ink() { bash "$R/scripts/ink_lium.sh" "$@"; }
bash $R/scripts/suite/push_pod.sh >/dev/null
ink scp ink-suite-1 $R/harness/build_cand.py /srv/suite/build_cand.py >/dev/null
for pair in $(python3 -c "import json;[print(f['name']+'='+f['cand_dir']) for f in json.load(open('$R/suite/finalists.json'))]"); do
  id=${pair%%=*}; d=$R/${pair#*=}
  ink exec ink-suite-1 "mkdir -p /srv/cands/$id" >/dev/null
  for f in $d/*; do ink scp ink-suite-1 "$f" "/srv/cands/$id/$(basename $f)" >/dev/null; done
  ink exec ink-suite-1 "bash /srv/suite/prep_cand.sh $id 2>&1 | tail -3"
done
