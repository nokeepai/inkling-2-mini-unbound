#!/bin/bash
set -euo pipefail
P=/Users/nikshepsvn/.inkling-abl/compare_private
mkdir -p $P/data
mv $P/plain/sealed/test/harmful.jsonl $P/plain/sealed/test/borderline.jsonl $P/data/
rm -rf $P/plain
python3 -c "
import json,collections
for n in ('harmful','borderline'):
    rows=[json.loads(l) for l in open('$P/data/'+n+'.jsonl')]
    print(n,len(rows),sorted(rows[0].keys()),collections.Counter(r.get('category') for r in rows).most_common())
"
shasum -a 256 $P/data/*.jsonl
