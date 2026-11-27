"""Verify every training-task module: clean module + tests must pass 100% of asserts.

For each mNN_* directory: copy module.py as solution.py next to tests.py in a temp
dir, then run the tests line by line in a shared namespace (the same way the RL
runner does) under the tinker_lab venv python. Also checks file shape rules.
Exits non-zero if any module has a problem.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get("LAB_PYTHON", sys.executable)
LINE1 = "import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))"
LINE2 = "from solution import *"

# Runs inside the temp dir; prints one JSON object with per-assert results.
RUNNER = r'''
import json, sys
path = sys.argv[1]
lines = open(path).read().splitlines()
ns = {"__file__": path, "__name__": "__tests__"}
out = {"setup_error": None, "asserts": [], "stmt_errors": []}
try:
    exec(lines[0], ns)
    exec(lines[1], ns)
except Exception as e:
    out["setup_error"] = repr(e)
    print(json.dumps(out)); sys.exit(0)
for no, line in enumerate(lines[2:], start=3):
    if not line.strip():
        continue
    is_assert = line.lstrip().startswith("assert")
    try:
        exec(line, ns)
        ok, err = True, None
    except Exception as e:
        ok, err = False, repr(e)
    if is_assert:
        out["asserts"].append({"line": no, "ok": ok, "err": err})
    elif not ok:
        out["stmt_errors"].append({"line": no, "err": err})
print(json.dumps(out))
'''


def shape_problems(d):
    problems = []
    mod = open(os.path.join(d, "module.py")).read().splitlines()
    if not 60 <= len(mod) <= 160:
        problems.append(f"module.py has {len(mod)} lines (want 60-160)")
    tests = open(os.path.join(d, "tests.py")).read().splitlines()
    if len(tests) < 2 or tests[0] != LINE1 or tests[1] != LINE2:
        problems.append("tests.py header lines wrong")
    body = [t for t in tests[2:] if t.strip()]
    if not 12 <= len(body) <= 25:
        problems.append(f"tests.py has {len(body)} test lines (want 12-25)")
    if len(body) != len(tests) - 2:
        problems.append("tests.py has blank lines")
    for t in body:
        if t.startswith((" ", "\t")) or t.rstrip().endswith((":", "\\", "(", "[", "{", ",")):
            problems.append(f"multi-line or indented statement: {t[:40]}")
    try:
        meta = json.load(open(os.path.join(d, "meta.json")))
        if not meta.get("name") or not meta.get("description"):
            problems.append("meta.json missing name/description")
    except (OSError, ValueError) as e:
        problems.append(f"meta.json unreadable: {e}")
    return problems, len(mod)


def run_tests(d):
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(d, "module.py"), os.path.join(tmp, "solution.py"))
        shutil.copy(os.path.join(d, "tests.py"), os.path.join(tmp, "tests.py"))
        runner = os.path.join(tmp, "_runner.py")
        with open(runner, "w") as f:
            f.write(RUNNER)
        proc = subprocess.run([PYTHON, runner, os.path.join(tmp, "tests.py")],
                              cwd=tmp, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            return None, proc.stderr.strip()[-300:]
        return json.loads(proc.stdout.strip().splitlines()[-1]), None


def main():
    dirs = sorted(x for x in os.listdir(HERE) if x.startswith("m") and os.path.isdir(os.path.join(HERE, x)))
    bad = 0
    for name in dirs:
        d = os.path.join(HERE, name)
        problems, n_lines = shape_problems(d)
        result, err = run_tests(d)
        if result is None:
            problems.append(f"runner crashed: {err}")
            passed = total = 0
        else:
            if result["setup_error"]:
                problems.append(f"setup error: {result['setup_error']}")
            for s in result["stmt_errors"]:
                problems.append(f"line {s['line']} statement error: {s['err']}")
            asserts = result["asserts"]
            total = len(asserts)
            passed = sum(a["ok"] for a in asserts)
            for a in asserts:
                if not a["ok"]:
                    problems.append(f"line {a['line']} failed: {a['err']}")
            if total < 12:
                problems.append(f"only {total} asserts (want >= 12)")
        status = "OK  " if not problems else "FAIL"
        bad += bool(problems)
        print(f"{status} {name:28s} module={n_lines:3d} lines  asserts={passed}/{total}")
        for p in problems:
            print(f"       - {p}")
    print(f"{len(dirs) - bad}/{len(dirs)} modules OK")
    if len(dirs) != 24:
        print(f"expected 24 module dirs, found {len(dirs)}")
        bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
