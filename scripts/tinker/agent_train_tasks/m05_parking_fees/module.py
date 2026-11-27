"""Parking-lot fee calculator with grace period, hourly rate, daily cap and night discount."""

GRACE_MINUTES = 15
HOURLY_CENTS = 300
DAILY_CAP_CENTS = 2000
MINUTES_PER_DAY = 1440
NIGHT_START = 22 * 60
NIGHT_LENGTH = 8 * 60


def parse_hhmm(text):
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError("bad time")
    h, m = int(parts[0]), int(parts[1])
    if h < 0 or h > 23 or m < 0 or m > 59:
        raise ValueError("bad time")
    return h * 60 + m


def duration(entry, exit_, days=0):
    """Minutes parked; days counts how many midnights passed."""
    start = parse_hhmm(entry)
    end = parse_hhmm(exit_) + days * MINUTES_PER_DAY
    if end < start:
        raise ValueError("exit before entry")
    return end - start


def billable_hours(minutes):
    if minutes <= GRACE_MINUTES:
        return 0
    return (minutes + 59) // 60


def fee_for_minutes(minutes):
    full_days = minutes // MINUTES_PER_DAY
    rest = minutes - full_days * MINUTES_PER_DAY
    fee = full_days * DAILY_CAP_CENTS
    part = billable_hours(rest) * HOURLY_CENTS
    return fee + min(part, DAILY_CAP_CENTS)


def night_discount(entry, minutes):
    """True when the whole stay fits inside one 22:00-06:00 night window."""
    start = parse_hhmm(entry)
    into_window = (start - NIGHT_START) % MINUTES_PER_DAY
    return into_window + minutes <= NIGHT_LENGTH


class Lot:
    def __init__(self, capacity):
        self.capacity = capacity
        self.parked = {}
        self.revenue = 0

    def enter(self, plate, time):
        if plate in self.parked or len(self.parked) >= self.capacity:
            return False
        self.parked[plate] = time
        return True

    def leave(self, plate, time, days=0):
        entry = self.parked.pop(plate)
        mins = duration(entry, time, days)
        fee = fee_for_minutes(mins)
        if night_discount(entry, mins):
            fee = fee // 2
        self.revenue += fee
        return fee

    def free_spaces(self):
        return self.capacity - len(self.parked)


def busiest_hour(entries):
    """Hour of day (0-23) with the most arrivals; earliest hour wins ties."""
    counts = [0] * 24
    for t in entries:
        counts[parse_hhmm(t) // 60] += 1
    best = 0
    for h in range(1, 24):
        if counts[h] > counts[best]:
            best = h
    return best
