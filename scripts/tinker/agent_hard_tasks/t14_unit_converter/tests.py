import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

uc = standard_converter()
assert uc.convert(2, "km", "cm") == 200000, "2 km in cm"
assert uc.convert(50000, "cm", "km") == 0.5, "50000 cm in km"
assert uc.convert(1, "mile", "ft") == 5280.0, "1 mile in ft"
assert uc.convert(100, "C", "F") == 212.0, "100 C in F"
assert uc.convert(212, "F", "C") == 100.0, "212 F in C"
assert uc.convert(32, "F", "K") == 273.15, "32 F in K"
assert uc.convert(0, "K", "F") == -459.67, "0 K in F"
assert uc.convert(491.67, "R", "C") == 0.0, "491.67 R in C"
assert uc.convert_many([0, 100], "C", "R") == [491.67, 671.67], "C to R"
uc.define("yd", "ft", 3)
assert uc.convert(1, "yd", "in") == 36.0, "1 yd in in"
assert uc.reachable_units("yd") == ["cm", "ft", "in", "km", "m", "mile"], "units reachable from yd"
assert uc.best_unit(2500, "m", ["km", "m", "cm"]) == "km", "best unit for 2500 m"
