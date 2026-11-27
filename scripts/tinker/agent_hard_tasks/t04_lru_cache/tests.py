import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

c = LRUCache(2)
c.put("a", 1)
c.put("b", 2)
assert c.get("a") == 1, "hit on a"
c.put("c", 3)
assert sorted(c.keys()) == ["a", "c"], "keys after third insert"
c = LRUCache(2)
c.put("a", 1)
c.put("b", 2)
c.put("a", 10)
assert len(c) == 2 and c.peek("b") == 2 and c.peek("a") == 10, "contents after updating a"
c.put("c", 3)
assert c.keys() == ["a", "c"], "keys after update then insert"
assert c.stats.evictions == 1, "eviction count"
c = LRUCache(2)
c.put("x", 1)
c.put("y", 2)
assert c.pop("x") == 1, "pop value"
c.put("z", 3)
c.put("w", 4)
assert c.items() == [("z", 3), ("w", 4)], "items after pop and two inserts"
calls = []

@memoize(2)
def square(n):
    calls.append(n)
    return n * n

assert [square(n) for n in [2, 3, 2, 4, 3]] == [4, 9, 4, 16, 9], "memoized values"
assert calls == [2, 3, 4, 3], "underlying calls"
assert square.cache.stats.as_dict() == {"hits": 1, "misses": 4, "evictions": 2, "hit_rate": 0.2}, "memo stats"
c = LRUCache(3)
for k in "abc":
    c.put(k, k)
c.get("a")
c.resize(1)
assert c.keys() == ["a"] and c.evicted == ["b", "c"], "resize keeps the most recent"
