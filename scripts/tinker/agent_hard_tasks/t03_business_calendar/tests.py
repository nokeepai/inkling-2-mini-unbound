import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *
from datetime import date

mon, fri, sat, sun = date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 6), date(2024, 1, 7)
assert business_days(mon, sat) == 5, "Monday through Saturday"
assert business_days(mon, fri) == 5, "Monday through Friday"
assert business_days(sat, sun) == 0, "a weekend"
assert add_business_days(fri, 1) == date(2024, 1, 8), "Friday plus one business day"
holidays = {date(2024, 1, 8)}
assert add_business_days(fri, 1, holidays) == date(2024, 1, 9), "Friday plus one with Monday off"
assert add_business_days(date(2024, 1, 10), -3) == date(2024, 1, 5), "three business days back"
assert overlap_days(mon, fri, fri, sun) == 1, "ranges sharing one day"
assert month_calendar(2024, 2)[-1] == [26, 27, 28, 29, 0, 0, 0], "last week of February 2024"
s = Schedule(holidays)
assert s.book("audit", sat, 3) == (date(2024, 1, 9), date(2024, 1, 11)), "booking from a Saturday"
s.book("review", date(2024, 1, 11), 2)
assert s.conflicts() == [("audit", "review")], "conflicting bookings"
assert s.free_days(mon, date(2024, 1, 16)) == [mon, date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), fri, date(2024, 1, 15), date(2024, 1, 16)], "free days"
