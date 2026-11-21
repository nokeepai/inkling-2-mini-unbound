"""Unit check of sgl_eval.loop_detect without loading the tokenizer (copies the function source)."""
import ast
import os
import random

src = open(os.path.expanduser("~/Code/inkling-abl/harness/sgl_eval.py")).read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "loop_detect")
ns: dict = {}
exec(compile(ast.Module([fn], []), "loop_detect", "exec"), ns)
ld = ns["loop_detect"]
rng = random.Random(0)
rand = lambda n: [rng.randrange(50000) for _ in range(n)]  # noqa: E731
span = rand(40)
cases = {
    "random 3000": (rand(3000), 0),
    "span40 x3 back to back": (rand(100) + span * 3 + rand(100), 1),
    "span40 x2 only": (rand(100) + span * 2 + rand(100), 0),
    "span40 x3 with gaps": (rand(50) + span + rand(5) + span + rand(5) + span + rand(50), 0),
    "span31 x3 (too short)": (rand(50) + span[:31] * 3 + rand(50), 0),
    "period8 x12 (=96 tokens)": (rand(50) + span[:8] * 12 + rand(50), 1),
    "period8 x11 (=88 tokens)": (rand(50) + span[:8] * 11 + rand(50), 0),
    "span200 x3": (rand(10) + rand(1) + (s := rand(200)) * 3, 1),
    "empty": ([], 0),
}
ok = True
for name, (ids, want) in cases.items():
    got = ld(ids)
    ok &= got == want
    print(f"{'OK ' if got == want else 'BAD'} {name}: {got} (want {want})")
print("ALL OK" if ok else "FAILURES")
