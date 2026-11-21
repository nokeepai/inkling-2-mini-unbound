"""One-off: fix manifest file list and teach card.py about SORRY-Bench."""
import json

p = '/Users/nikshepsvn/Code/inkling-abl/sealed/raw/manifest.json'
m = json.load(open(p))
m['sorrybench']['files'] = ['sorrybench/question.jsonl', 'sorrybench/meta_info.py']
m['sorrybench']['errors'] = []
json.dump(m, open(p, 'w'), indent=1)

p = '/Users/nikshepsvn/Code/inkling-abl/scripts/data/card.py'
s = open(p).read()
s = s.replace('"sorrybench": "custom (gated; see repo LICENSE)",', '"sorrybench": "custom, gated (see repo LICENSE; card says other)",')
s = s.replace('"sorrybench": "harmful: SKIPPED",', '"sorrybench": "harmful (question.jsonl, 440 base prompts; 80 in 8 excluded categories; 20 mutation files not used)",')
old = s[s.index('src.append("SORRY-Bench: both'):s.index('(R / "harness" / "SOURCES.md")')]
s = s.replace(old, 'src.append("SORRY-Bench: the gate was accepted by the account owner on 2026-11-21; 202503 pinned. Only `question.jsonl` (prompt_style == base) is used; category names from the repo\'s `meta_info.py`.\\n")\n')
s = s.replace('''        "| violence | | Violence |", "| sexual | | Sexual content |", "| other_harm | harmful | |\\n",''', '''        "| violence | | Violence |", "| sexual | | Sexual content |", "| other_harm | harmful | |\\n",
        "### SORRY-Bench 202503 (44 categories -> coarse; `orig_category` = `<id>:<short name>`)",
        "| id | SORRY-Bench category | coarse |", "|---|---|---|"] + [
        f"| {i} | {n} | {c or 'EXCLUDED (not harmful to answer: opinion promotion or specialised advice)'} |" for i, (n, c) in SORRY_MAP.items()] + ["",''')
s = s.replace('from pathlib import Path\n', 'from pathlib import Path\nimport sys\nsys.path.insert(0, "/Users/nikshepsvn/Code/inkling-abl/scripts/data")\nfrom build import SORRY_MAP  # noqa: E402\n', 1)
s = s.replace('priority order: probes, xstest, orbench, harmbench, strongreject, advbench,', 'priority order: probes, xstest, orbench, harmbench, strongreject, sorrybench, advbench,')
s = s.replace('"5. Harmful categories: HarmBench and StrongREJECT labels', '"5. Harmful categories: HarmBench, StrongREJECT and SORRY-Bench labels')
s = s.replace('rows = S["raw"].get(k, "")', 'rows = S["raw"].get(k, "")\n    if k == "sorrybench":\n        rows = f"440 ({S[\'raw\'].get(k, 0)} after category exclusion)"')
open(p, 'w').write(s)
print('SORRY_MAP' in s, 'SKIPPED' in s, 'sorrybench, advbench' in s)
