"""Business-day calendar helpers built on datetime.date, plus a booking schedule."""
from datetime import date, timedelta

ONE_DAY = timedelta(days=1)


def parse_date(text):
    year, month, day = (int(part) for part in text.split("-"))
    return date(year, month, day)


def daterange(start, end):
    """Every date from start to end, both inclusive."""
    if end < start:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days)]


def is_weekend(day):
    return day.weekday() > 5


def is_business_day(day, holidays=()):
    return not is_weekend(day) and day not in holidays


def business_days(start, end, holidays=()):
    """Number of business days between start and end, both inclusive."""
    return sum(1 for day in daterange(start, end) if is_business_day(day, holidays))


def add_business_days(start, n, holidays=()):
    """Date n business days after start (negative n goes backwards; start is not counted)."""
    day = start
    step = ONE_DAY if n >= 0 else -ONE_DAY
    remaining = abs(n)
    while remaining > 0:
        day += step
        if is_business_day(day, holidays):
            remaining -= 1
    return day


def next_business_day(day, holidays=()):
    return add_business_days(day, 1, holidays)


def overlap_days(a_start, a_end, b_start, b_end):
    """Calendar days shared by two inclusive ranges."""
    return len(daterange(max(a_start, b_start), min(a_end, b_end)))


def month_calendar(year, month):
    """Weeks of the month as lists of day numbers, Monday first, 0 for padding."""
    first = date(year, month, 1)
    following = date(year + (month == 12), month % 12 + 1, 1)
    weeks, week = [], [0] * first.weekday()
    for day in daterange(first, following - ONE_DAY):
        week.append(day.day)
        if len(week) == 7:
            weeks.append(week)
            week = []
    if week:
        weeks.append(week + [0] * (7 - len(week)))
    return weeks


def format_range(start, end):
    if start == end:
        return start.isoformat()
    return f"{start.isoformat()}..{end.isoformat()}"


class Schedule:
    """Bookings measured in business days."""

    def __init__(self, holidays=()):
        self.holidays = set(holidays)
        self.bookings = {}

    def book(self, name, start, length):
        """Book `length` business days beginning on the first business day >= start."""
        if length < 1:
            raise ValueError("length must be at least 1")
        first = start if is_business_day(start, self.holidays) else next_business_day(start, self.holidays)
        last = add_business_days(first, length - 1, self.holidays)
        self.bookings[name] = (first, last)
        return first, last

    def conflicts(self):
        names = sorted(self.bookings)
        found = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if overlap_days(*self.bookings[a], *self.bookings[b]) > 0:
                    found.append((a, b))
        return found

    def free_days(self, start, end):
        """Business days in [start, end] not covered by any booking."""
        taken = set()
        for first, last in self.bookings.values():
            taken.update(daterange(first, last))
        return [d for d in daterange(start, end)
                if is_business_day(d, self.holidays) and d not in taken]

    def describe(self):
        return [f"{name}: {format_range(*self.bookings[name])}" for name in sorted(self.bookings)]
