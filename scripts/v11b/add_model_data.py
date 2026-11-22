"""Mac: add/refresh one model's sealed-test (cap 8192) numbers in package_v2/data_test8k.json from its scored dir.
  python3 add_model_data.py <key> <scored dir name under v11b_private/val/out>"""
import collections, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'compare'))
import aggregate

key, name = sys.argv[1:3]
P = os.path.expanduser('~/.inkling-abl/v11b_private/val/out')
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'package_v2', 'data_test8k.json')
d = json.load(open(p)); m = {}
for mode in ('off', 'e09'):
    G = {r['id']: r for r in map(json.loads, open(f'{P}/{name}/gens_harmful_{mode}.jsonl'))}
    J = [json.loads(l) for l in open(f'{P}/{name}/jev_harmful_{mode}.jsonl')]
    u = [float(r['useful']) for r in J]
    m[f'useful_{mode}'] = round(100 * sum(u) / len(u), 1)
    m[f'useful_{mode}_ci'] = [round(100 * x, 1) for x in aggregate.boot(u)]
    m[f'refusal_{mode}'] = round(100 * sum(r['refusal'] for r in G.values()) / len(G), 1)
    m[f'thinkcap_{mode}'] = round(100 * sum(r['cap_hit'] for r in G.values()) / len(G), 1)
    cat = collections.defaultdict(list)
    for r in J: cat[G[r['id']]['category']].append(float(r['useful']))
    m[f'useful_{mode}_by_cat'] = {c: round(100 * sum(v) / len(v), 1) for c, v in sorted(cat.items())}
    m['n_by_cat'] = {c: len(v) for c, v in sorted(cat.items())}
d['models'][key] = m
v2 = {mode: {r['id']: float(r['useful']) for r in map(json.loads, open(f'{P}/test8k_E/jev_harmful_{mode}.jsonl'))} for mode in ('off', 'e09')}
for mode in ('off', 'e09'):
    o = {r['id']: float(r['useful']) for r in map(json.loads, open(f'{P}/{name}/jev_harmful_{mode}.jsonl'))}
    A = v2[mode]
    d['head_to_head'][f'v2_vs_{key}_{mode}'] = {'v2_only': sum(1 for i in A if A[i] and not o[i]), 'other_only': sum(1 for i in A if o[i] and not A[i]),
                                               'both': sum(1 for i in A if A[i] and o[i]), 'neither': sum(1 for i in A if not A[i] and not o[i])}
json.dump(d, open(p, 'w'), indent=1)
print(key, m['useful_off'], m['useful_e09'], m['refusal_off'], m['refusal_e09'])
