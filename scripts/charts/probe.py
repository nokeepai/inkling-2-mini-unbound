import json
d = json.load(open('/Users/nikshepsvn/Code/inkling-abl/compare/results.json'))
for k, v in d.items():
    m = v.get('meta', {})
    u = v.get('jev_useful_e09')
    print(k, '|', m.get('label'), '|', m.get('precision'), '|', m.get('setup', '')[:70], '|',
          u and (round(u['value'] * 100, 1), [round(x * 100, 1) for x in u['CI']], u['n']))
