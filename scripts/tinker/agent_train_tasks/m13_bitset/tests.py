import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import *
b = BitSet(8)
assert b.count() == 0, "empty count"
assert b.first_clear() == 0, "first clear of empty"
b.set(0)
b.set(7)
b.set_range(2, 5)
assert b.members() == [0, 2, 3, 4, 7], "members"
assert b.to_string() == "10011101", "to string msb first"
assert b.count() == 5, "count"
assert b.test(1) == False, "unset bit"
assert b.first_clear() == 1, "first clear"
assert b.longest_run() == 3, "longest run"
b.toggle(3)
assert b.test(3) == False, "toggle off"
b.clear(0)
assert b.members() == [2, 4, 7], "after clear"
c = from_members(4, [1, 2])
assert union(b, c).members() == [1, 2, 4, 7], "union"
assert intersection(b, c).members() == [2], "intersection"
assert difference(b, c).members() == [4, 7], "difference"
full = from_members(3, [0, 1, 2])
assert full.first_clear() == -1, "no clear bit"
assert full.to_string() == "111", "full string"
assert full.longest_run() == 3, "full run"
