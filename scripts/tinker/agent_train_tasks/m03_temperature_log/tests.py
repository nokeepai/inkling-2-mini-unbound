import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import *
log = TempLog()
assert log.daily_range() == 0, "empty range"
assert log.hottest_hour() is None, "empty hottest"
log.record(6, -2)
log.record(0, 1)
log.record(12, 15)
log.record(18, 15)
assert log.values() == [1, -2, 15, 15], "sorted by hour"
assert log.daily_range() == 17, "daily range"
assert log.hottest_hour() == 12, "earliest hottest on tie"
assert log.above(1) == 2, "strictly above"
assert frost_hours(log) == [6], "frost at zero"
assert frost_hours(log, 1) == [0, 6], "frost inclusive limit"
assert to_fahrenheit(100) == 212, "boiling"
assert longest_warming_streak([]) == 0, "streak empty"
assert longest_warming_streak([1, 2, 3, 3, 4, 5, 6]) == 4, "streak reset on equal"
assert longest_warming_streak([5, 4, 3]) == 1, "no warming"
assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5], "moving average"
assert moving_average([1, 2], 3) == [], "window too big"
assert moving_average([1, 2, 3], 3) == [2.0], "window equals length"
assert degree_hours([10, 18, 20, 17]) == 9, "degree hours"
assert biggest_jump([1, -2, 15, 15]) == 17, "biggest jump"
assert comfort_band([17, 18, 24, 25]) == (1, 2, 1), "comfort band boundaries"
