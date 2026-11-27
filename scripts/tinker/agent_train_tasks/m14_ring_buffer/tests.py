import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import *
r = RingBuffer(3)
assert r.to_list() == [], "empty list"
assert r.average() == 0.0, "empty average"
assert r.is_full() == False, "not full"
r.extend([1, 2, 3])
assert r.is_full() == True, "full at capacity"
assert r.to_list() == [1, 2, 3], "contents"
r.push(4)
assert r.to_list() == [2, 3, 4], "oldest overwritten"
assert r.dropped == 1, "dropped count"
assert r.peek() == 2, "peek oldest"
assert r.newest() == 4, "newest"
assert r.pop() == 2, "pop oldest"
assert r.to_list() == [3, 4], "after pop"
r.push(5)
assert r.to_list() == [3, 4, 5], "wraparound push"
assert r.last_n(2) == [4, 5], "last two"
assert r.last_n(0) == [], "last zero"
assert r.last_n(9) == [3, 4, 5], "last n clamps"
assert r.average() == 4.0, "average"
assert rolling_max([1, 5, 2, 1, 1], 2) == [1, 5, 5, 2, 1], "rolling max"
assert rolling_sum_exceeds([3, 3, 1, 5], 2, 5) == [1, 3], "rolling sum exceeds"
one = RingBuffer(1)
one.extend([7, 8])
assert one.to_list() == [8], "capacity one"
