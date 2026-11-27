"""Histogram binning, percentiles and text rendering."""


def bin_edges(lo, hi, bins):
    if bins <= 0 or hi <= lo:
        raise ValueError("bad range")
    width = (hi - lo) / bins
    return [lo + i * width for i in range(bins + 1)]


def bin_index(value, lo, hi, bins):
    """Index of the bin holding value; the top edge belongs to the last bin."""
    if value < lo or value > hi:
        return -1
    if value == hi:
        return bins - 1
    return int((value - lo) * bins // (hi - lo))


def histogram(values, lo, hi, bins):
    counts = [0] * bins
    for v in values:
        idx = bin_index(v, lo, hi, bins)
        if idx >= 0:
            counts[idx] += 1
    return counts


def outliers(values, lo, hi):
    below = sum(1 for v in values if v < lo)
    above = sum(1 for v in values if v > hi)
    return below, above


def mode_bin(counts):
    """Fullest bin; the lowest index wins ties."""
    best = 0
    for i in range(1, len(counts)):
        if counts[i] > counts[best]:
            best = i
    return best if counts else -1


def cumulative(counts):
    out = []
    running = 0
    for c in counts:
        running += c
        out.append(running)
    return out


def percentile_bin(counts, pct):
    """First bin at which the cumulative count reaches pct percent of the total."""
    total = sum(counts)
    if total == 0:
        return -1
    need = total * pct / 100
    running = 0
    for i in range(len(counts)):
        running += counts[i]
        if running >= need:
            return i
    return len(counts) - 1


def render(counts, width=10, char="#"):
    """Bars scaled so the peak is width long; any non-zero count gets at least one char."""
    peak = max(counts) if counts else 0
    rows = []
    for c in counts:
        bar = 0 if peak == 0 else (c * width + peak - 1) // peak
        rows.append(char * bar)
    return rows


def merge_bins(counts, factor):
    if factor <= 0:
        raise ValueError("factor must be positive")
    return [sum(counts[i:i + factor]) for i in range(0, len(counts), factor)]


def empty_runs(counts):
    """Length of the longest run of consecutive empty bins."""
    best = run = 0
    for c in counts:
        run = run + 1 if c == 0 else 0
        best = max(best, run)
    return best
