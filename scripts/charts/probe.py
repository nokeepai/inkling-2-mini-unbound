import json
for l in open('/Users/nikshepsvn/Code/inkling-abl/audit/test/kl_results.jsonl'):
    r = json.loads(l)
    r.pop('per_prompt_kl', None)
    r.pop('pos_stats', None)
    print(r)
