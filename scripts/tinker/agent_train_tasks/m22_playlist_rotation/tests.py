import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import *
assert rotate([1, 2, 3, 4], 1) == [2, 3, 4, 1], "rotate left"
assert rotate([1, 2, 3], -1) == [3, 1, 2], "rotate right"
assert rotate([1, 2, 3], 3) == [1, 2, 3], "full rotation"
assert rotate([], 5) == [], "rotate empty"
assert interleave([1, 2, 3], ["a"]) == [1, "a", 2, 3], "interleave uneven"
assert riffle([1, 2, 3, 4, 5]) == [1, 4, 2, 5, 3], "riffle odd"
assert riffle([]) == [], "riffle empty"
assert stride_order(5, 2) == [0, 2, 4, 1, 3], "stride coprime"
assert stride_order(4, 2) == [0, 2, 1, 3], "stride with collision"
assert stride_order(3, 0) == [0, 1, 2], "zero stride"
assert stride_order(0, 3) == [], "stride empty"
p = Playlist(["a", "b", "c", "d", "e"])
assert p.current() == "a", "current"
assert p.prev() == "e", "prev wraps"
assert p.next() == "a", "next wraps"
assert p.window(3) == ["a", "b", "c"], "window"
assert p.window(9) == ["a", "b", "c", "d", "e"], "window clamps"
p.shuffle(2)
assert p.songs == ["e", "b", "d", "a", "c"], "deterministic shuffle"
p.move(0, 4)
assert p.songs == ["b", "d", "a", "c", "e"], "move song"
assert Playlist([]).current() is None, "empty playlist"
assert no_repeat_artist([("x", 1), ("y", 2), ("x", 3)]) == True, "no repeats"
assert no_repeat_artist([("x", 1), ("x", 2)]) == False, "back to back artist"
