import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import *
assert gcd(12, 18) == 6, "gcd"
assert simplify(6, 8) == (3, 4), "simplify"
assert simplify(3, -6) == (-1, 2), "negative denominator"
assert simplify(0, 5) == (0, 1), "zero numerator"
assert scale(1, 2, 4, 8) == (1, 1), "double recipe"
assert scale(3, 4, 4, 2) == (3, 8), "halve recipe"
assert format_fraction(7, 2) == "3 1/2", "mixed number"
assert format_fraction(1, 3) == "1/3", "proper fraction"
assert format_fraction(4, 2) == "2", "whole"
assert format_fraction(0, 3) == "0", "zero"
assert to_tsp(1, 2, "cup") == (24, 1), "half cup in tsp"
assert best_unit(48, 1) == ("cup", (1, 1)), "exactly one cup"
assert best_unit(47, 1) == ("tbsp", (47, 3)), "just under a cup"
assert best_unit(2, 1) == ("tsp", (2, 1)), "teaspoons"
assert best_unit(3, 1) == ("tbsp", (1, 1)), "exactly one tbsp"
r = Recipe(4)
r.add("flour", 2, 1, "cup")
r.add("salt", 1, 2, "tsp")
assert r.scaled(2) == [("flour", "1", "cup"), ("salt", "1/4", "tsp")], "scale down"
assert r.scaled(6) == [("flour", "3", "cup"), ("salt", "3/4", "tsp")], "scale up"
assert r.total_tsp() == (193, 2), "total tsp"
assert Recipe(1).total_tsp() == (0, 1), "empty recipe total"
assert servings_for(3, 1, 2, 1, 4) == 6, "servings from available"
assert servings_for(1, 1, 2, 1, 1) == 0, "not enough for one serving"
