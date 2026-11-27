import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

assert evaluate("1 + 2 * 3") == 7, "1 + 2 * 3"
assert evaluate("(1 + 2) * -3") == -9, "(1 + 2) * -3"
assert evaluate("10 - 3 - 2") == 5, "10 - 3 - 2"
assert evaluate("20 / 2 / 5") == 2, "20 / 2 / 5"
assert evaluate("2 ^ 3 ^ 2") == 512, "2 ^ 3 ^ 2"
assert evaluate("2 ^ -1") == 0.5, "2 ^ -1"
assert evaluate("-2 ^ 2") == -4, "-2 ^ 2"
assert evaluate("max(1, 8 - 2 - 1, 3) + min(4, 2)") == 7, "max(1, 8 - 2 - 1, 3) + min(4, 2)"
assert evaluate("x * 2 - y", {"x": 4, "y": 1}) == 7, "x * 2 - y with x=4, y=1"
assert evaluate("17 % 5 * 2") == 4, "17 % 5 * 2"
assert evaluate_all(["a = 3", "b = a ^ 2 - 4 - 1", "c = -b ^ 0.5"]) == {"a": 3, "b": 4, "c": -2}, "chained assignments"
