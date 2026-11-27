"""Single-turn debugging tasks for w3_code: AST-mutated copies of agent_train_tasks modules.

Each task = a correct module from agent_train_tasks/ with 2 injected bugs (comparison flips,
+/- swaps, and/or swaps, integer constant +-1), kept only if it still imports and passes
<= 80% of the module's asserts. Reward = fraction of asserts passing after the model's fix.
The 24 source modules are disjoint from agent_eval_hard's 15.

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/debugpool.py   # -> prompts_train_code.jsonl
"""
from __future__ import annotations

import ast
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import lab_common as lc

SRC = Path(__file__).parent / "agent_train_tasks"
SEED = 2026
PER_MODULE = 7
RUNNER = """
import sys, json
sys.path.insert(0, '.')
lines = open('tests.py').read().splitlines()[2:]
ns = {}
exec('from solution import *', ns)
passed = total = 0
for line in lines:
    is_assert = line.lstrip().startswith('assert')
    total += is_assert
    try:
        exec(line, ns)
        passed += is_assert
    except Exception:
        pass
print(json.dumps({'passed': passed, 'total': total}))
"""


def pass_fraction(code: str, tests: str) -> float:
    """Fraction of assert lines passing; 0.0 if the module fails to import or times out."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "solution.py").write_text(code)
        (Path(d) / "tests.py").write_text(tests)
        (Path(d) / "runner.py").write_text(RUNNER)
        try:
            res = subprocess.run([sys.executable, "-I", "runner.py"], cwd=d, capture_output=True,
                                 text=True, timeout=10)
            out = json.loads(res.stdout.strip().splitlines()[-1])
            return out["passed"] / max(1, out["total"])
        except Exception:
            return 0.0


class Mutator(ast.NodeTransformer):
    """Applies the k-th eligible mutation only, so each call injects exactly one bug."""
    SWAPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
             ast.Add: ast.Sub, ast.Sub: ast.Add, ast.And: ast.Or, ast.Or: ast.And}

    def __init__(self, target: int) -> None:
        self.target, self.seen = target, 0

    def _hit(self) -> bool:
        self.seen += 1
        return self.seen - 1 == self.target

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if type(node.ops[0]) in self.SWAPS and self._hit():
            node.ops[0] = self.SWAPS[type(node.ops[0])]()
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if type(node.op) in self.SWAPS and self._hit():
            node.op = self.SWAPS[type(node.op)]()
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if type(node.op) in self.SWAPS and self._hit():
            node.op = self.SWAPS[type(node.op)]()
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, int) and not isinstance(node.value, bool) and self._hit():
            return ast.Constant(node.value + random.choice([-1, 1]))
        return node


def count_sites(tree: ast.AST) -> int:
    m = Mutator(-1)
    m.visit(tree)
    return m.seen


def mutate(code: str, rng: random.Random, n_bugs: int = 2) -> str:
    for _ in range(n_bugs):
        tree = ast.parse(code)
        tree = Mutator(rng.randrange(count_sites(ast.parse(code)))).visit(tree)
        code = ast.unparse(tree) + "\n"
    return code


def prompt(desc: str, buggy: str) -> str:
    return (f"This module ({desc}) has bugs. I have a hidden unit-test suite for it.\n\n"
            f"```python\n{buggy}```\n\nFix the bugs. Reply with the COMPLETE corrected module "
            "in a single ```python code block.")


def main() -> None:
    rng = random.Random(SEED)
    random.seed(SEED)
    rows: list[dict[str, Any]] = []
    for d in sorted(SRC.glob("m[0-9][0-9]_*")):
        code, tests = (d / "module.py").read_text(), (d / "tests.py").read_text()
        desc = json.loads((d / "meta.json").read_text())["description"]
        # The model sees ast.unparse'd code, so the clean reference must match its formatting.
        assert pass_fraction(ast.unparse(ast.parse(code)) + "\n", tests) == 1.0, d.name
        made, tries = 0, 0
        while made < PER_MODULE and tries < 60:
            tries += 1
            buggy = mutate(code, rng)
            frac = pass_fraction(buggy, tests)
            if frac <= 0.8:
                rows.append({"id": f"dbg{len(rows):04d}", "category": "debug", "module": d.name,
                             "prompt": prompt(desc, buggy), "tests": tests, "buggy_pass": round(frac, 3),
                             "target_words": None, "gold": None})
                made += 1
    base = lc.read_jsonl(lc.LAB_DIR / "prompts_train_hard.jsonl")
    lc.write_jsonl(lc.LAB_DIR / "prompts_train_code.jsonl", base + rows)
    fr = [r["buggy_pass"] for r in rows]
    print(f"debug rows={len(rows)} base={len(base)} share={len(rows) / (len(rows) + len(base)):.2f} "
          f"mean buggy pass={sum(fr) / len(fr):.2f}")


if __name__ == "__main__":
    main()
