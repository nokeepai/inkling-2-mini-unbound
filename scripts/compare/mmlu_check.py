import collections
import json
import os
for m in sorted(os.listdir('/root/compare/out')):
    p = f'/root/compare/out/{m}/gens_mmlu.jsonl'
    if not os.path.exists(p):
        continue
    r = [json.loads(l) for l in open(p)]
    acc = sum(x['pred'] == x['answer'] for x in r) / len(r)
    print(m, round(acc * 100, 1), 'missing', sum(x['missing'] for x in r), 'pred', sorted(collections.Counter(x['pred'] for x in r).items()),
          'maxlp_mean', round(sum(max(x['lp']) for x in r) / len(r), 3))
