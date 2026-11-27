"""Timesheet and payroll calculator with daily/weekly overtime and tax brackets."""

REGULAR_DAILY = 8 * 60
WEEKLY_REGULAR = 40 * 60
OVERTIME_NUM = 3
OVERTIME_DEN = 2
# weekly gross upper bounds in cents with their marginal tax percent
BRACKETS = [(50000, 10), (150000, 20), (None, 30)]


def minutes_between(start, end):
    """Minutes from start to end clock times; an earlier end means the next day."""
    sh, sm = start.split(":")
    eh, em = end.split(":")
    s = int(sh) * 60 + int(sm)
    e = int(eh) * 60 + int(em)
    if e < s:
        e += 24 * 60
    return e - s


def shift_minutes(start, end, break_minutes=0):
    worked = minutes_between(start, end) - break_minutes
    return max(0, worked)


def split_week(daily_minutes):
    """Return (regular, overtime): time past 8h a day or past 40h a week is overtime."""
    regular = 0
    overtime = 0
    for m in daily_minutes:
        day_regular = min(m, REGULAR_DAILY)
        overtime += m - day_regular
        room = WEEKLY_REGULAR - regular
        if day_regular > room:
            overtime += day_regular - room
            day_regular = room
        regular += day_regular
    return regular, overtime


def gross_pay(daily_minutes, hourly_cents):
    regular, overtime = split_week(daily_minutes)
    base = regular * hourly_cents // 60
    extra = overtime * hourly_cents * OVERTIME_NUM // (60 * OVERTIME_DEN)
    return base + extra


def income_tax(gross):
    tax = 0
    lower = 0
    for upper, pct in BRACKETS:
        if upper is None or gross <= upper:
            tax += (gross - lower) * pct // 100
            return tax
        tax += (upper - lower) * pct // 100
        lower = upper
    return tax


def net_pay(daily_minutes, hourly_cents):
    gross = gross_pay(daily_minutes, hourly_cents)
    return gross - income_tax(gross)


def format_hours(minutes):
    return f"{minutes // 60}:{minutes % 60:02d}"


def week_from_shifts(shifts):
    """shifts: list of (start, end, break_minutes) tuples, one per day."""
    return [shift_minutes(s, e, b) for s, e, b in shifts]


def effective_rate(daily_minutes, hourly_cents):
    """Net cents per hour worked, rounded down; 0 when nothing was worked."""
    worked = sum(daily_minutes)
    if worked == 0:
        return 0
    return net_pay(daily_minutes, hourly_cents) * 60 // worked
