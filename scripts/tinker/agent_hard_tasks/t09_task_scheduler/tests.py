import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

s = Scheduler()
s.add("fetch", priority=2)
s.add("lint", priority=1)
s.add("build", priority=1, deps=["fetch"])
s.add("docs", priority=1)
assert s.order() == ["lint", "docs", "fetch", "build"], "order of fetch/lint/build/docs"
assert s.ready({"fetch"}) == ["lint", "build", "docs"], "ready after fetch"
s = Scheduler()
s.add("a", priority=5)
s.add("b", priority=5, deps=["a"])
s.add("c", priority=0, deps=["a", "b"])
s.add("d", priority=1)
assert s.order() == ["d", "a", "b", "c"], "order of a/b/c/d"
s = Scheduler()
s.add("x", deps=["z"])
s.add("y", deps=["x"])
s.add("z", deps=["y"])
s.add("w")
try:
    s.order()
    detected = False
except CycleError:
    detected = True
assert detected, "x -> z -> y -> x"
s = Scheduler()
for name, deps in [("base", []), ("mid", ["base"]), ("top", ["mid"]), ("side", []), ("leaf", ["side"])]:
    s.add(name, deps=deps)
assert s.cancel("base") == ["base", "mid", "top"], "cancel base"
assert s.order() == ["side", "leaf"], "order after cancel"
s = from_spec(["a 0 3", "b 0 2", "c 0 1 a", "d 0 4 b"])
assert s.makespan(workers=2) == 6, "makespan with two workers"
assert s.makespan(workers=1) == 10, "makespan with one worker"
assert s.critical_path() == 6, "critical path"
