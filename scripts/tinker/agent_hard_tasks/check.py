"""Sanity-check every hard debugging task.

For each tNN_* directory: buggy.py must FAIL tests.py, fixed.py must PASS, and every
buggy_partial_<k>.py (all bugs present except bug k) must still FAIL, so no single fix
is enough. Prints one OK/FAIL line per task and exits non-zero on any problem.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT_S = 10


def run_candidate(task_dir: Path, candidate: Path) -> tuple[bool, str]:
    """Run tests.py against candidate saved as solution.py; return (passed, last output line)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy(candidate, tmp_path / "solution.py")
        shutil.copy(task_dir / "tests.py", tmp_path / "tests.py")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "tests.py"], cwd=tmp_path,
                capture_output=True, text=True, timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        output = (proc.stdout + proc.stderr).strip().splitlines()
        return proc.returncode == 0, output[-1] if output else ""


def check_task(task_dir: Path) -> list[str]:
    problems: list[str] = []
    for name in ("buggy.py", "fixed.py", "tests.py", "meta.json"):
        if not (task_dir / name).exists():
            problems.append(f"missing {name}")
    if problems:
        return problems
    meta = json.loads((task_dir / "meta.json").read_text())
    n_lines = len((task_dir / "buggy.py").read_text().splitlines())
    if not 100 <= n_lines <= 300:
        problems.append(f"buggy.py has {n_lines} lines")
    if not 2 <= len(meta.get("bugs", [])) <= 3:
        problems.append("meta.json should list 2-3 bugs")
    passed, line = run_candidate(task_dir, task_dir / "fixed.py")
    if not passed:
        problems.append(f"fixed.py fails: {line}")
    passed, _ = run_candidate(task_dir, task_dir / "buggy.py")
    if passed:
        problems.append("buggy.py passes")
    partials = sorted(task_dir.glob("buggy_partial_*.py"))
    if len(partials) != len(meta.get("bugs", [])):
        problems.append(f"{len(partials)} partial variants for {len(meta.get('bugs', []))} bugs")
    for partial in partials:
        passed, _ = run_candidate(task_dir, partial)
        if passed:
            problems.append(f"{partial.name} passes (one fix is enough)")
    return problems


def main() -> int:
    task_dirs = sorted(p for p in HERE.iterdir() if p.is_dir() and p.name[:1] == "t" and p.name[1:3].isdigit())
    failures = 0
    for task_dir in task_dirs:
        problems = check_task(task_dir)
        status = "OK  " if not problems else "FAIL"
        failures += bool(problems)
        print(f"{status} {task_dir.name}" + ("" if not problems else "  " + "; ".join(problems)))
    print(f"{len(task_dirs) - failures}/{len(task_dirs)} tasks OK")
    return 1 if failures or not task_dirs else 0


if __name__ == "__main__":
    sys.exit(main())
