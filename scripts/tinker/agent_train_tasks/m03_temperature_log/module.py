"""Temperature log analytics over hourly readings in Celsius."""


class TempLog:
    def __init__(self):
        self.readings = []

    def record(self, hour, celsius):
        if hour < 0 or hour > 23:
            raise ValueError("hour out of range")
        self.readings.append((hour, celsius))
        self.readings.sort()

    def values(self):
        return [c for _, c in self.readings]

    def daily_range(self):
        vals = self.values()
        if not vals:
            return 0
        return max(vals) - min(vals)

    def hottest_hour(self):
        """Earliest hour with the maximum reading."""
        best_hour = None
        best = None
        for hour, c in self.readings:
            if best is None or c > best:
                best = c
                best_hour = hour
        return best_hour

    def above(self, threshold):
        return sum(1 for _, c in self.readings if c > threshold)


def to_fahrenheit(celsius):
    return celsius * 9 / 5 + 32


def longest_warming_streak(values):
    """Length of the longest run of strictly increasing consecutive readings."""
    if not values:
        return 0
    best = 1
    run = 1
    for i in range(1, len(values)):
        if values[i] > values[i - 1]:
            run += 1
            if run > best:
                best = run
        else:
            run = 1
    return best


def moving_average(values, window):
    if window <= 0 or window > len(values):
        return []
    out = []
    for i in range(len(values) - window + 1):
        out.append(sum(values[i:i + window]) / window)
    return out


def frost_hours(log, limit=0):
    return [h for h, c in log.readings if c <= limit]


def degree_hours(values, base=18):
    """Heating degree-hours: total amount each reading falls below base."""
    total = 0
    for v in values:
        if v < base:
            total += base - v
    return total


def biggest_jump(values):
    best = 0
    for i in range(len(values) - 1):
        diff = abs(values[i + 1] - values[i])
        if diff > best:
            best = diff
    return best


def comfort_band(values, low=18, high=24):
    """Counts of readings below, inside (inclusive) and above the comfort band."""
    below = inside = above = 0
    for v in values:
        if v < low:
            below += 1
        elif v <= high:
            inside += 1
        else:
            above += 1
    return below, inside, above
