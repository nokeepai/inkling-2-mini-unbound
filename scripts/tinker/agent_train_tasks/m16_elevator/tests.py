import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import *
e = Elevator(10, 3)
assert e.call(3) == False, "current floor call ignored"
assert e.call(10) == False, "above top floor"
assert e.call(-1) == False, "below ground"
assert e.step() == False, "idle step"
assert e.call(5) == True, "call up"
assert e.call(1) == True, "call down"
assert e.choose_direction() == 1, "idle prefers up"
assert e.run() == 6, "steps to serve both"
assert e.visited == [5, 1], "visit order"
assert e.travelled == 6, "travelled"
assert e.direction == 0, "idle after run"
assert simulate(10, 5, [2, 8, 6]) == ([6, 8, 2], 9), "scan up then down"
assert simulate(5, 4, [0]) == ([0], 4), "go to ground"
assert simulate(5, 0, []) == ([], 0), "no calls"
assert total_distance(5, [6, 8, 2]) == 9, "total distance"
assert total_distance(0, []) == 0, "no stops distance"
assert fifo_distance(5, [2, 8, 6]) == 11, "fifo distance"
f = Elevator(10, 0)
f.call(4)
f.step()
f.step()
f.call(1)
assert f.run() == 5, "keeps direction before reversing"
assert f.visited == [4, 1], "visited after reversal"
