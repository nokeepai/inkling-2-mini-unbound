"""Interval utilities for closed intervals given as [start, end] pairs."""


def _ordered(interval):
    start, end = interval
    return (start, end) if start <= end else (end, start)


def merge(intervals):
    """Merge overlapping or touching intervals into a sorted list of new [start, end] lists."""
    spans = sorted(intervals, key=lambda iv: min(iv))
    merged = []
    for iv in spans:
        start, end = _ordered(iv)
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append(iv)
    return merged


def total_covered(intervals):
    return sum(end - start for start, end in merge(intervals))


def contains_point(intervals, point):
    return any(start <= point <= end for start, end in merge(intervals))


def gaps(intervals, lo, hi):
    """Uncovered stretches of [lo, hi] as (start, end) tuples."""
    out = []
    cursor = lo
    for start, end in merge(intervals):
        if end < lo or start > hi:
            continue
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < hi:
        out.append((cursor, hi))
    return out


def intersect(a, b):
    """Intersection of two interval lists."""
    a, b = merge(a), merge(b)
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo <= hi:
            out.append([lo, hi])
        if a[i][0] < b[j][0]:
            i += 1
        else:
            j += 1
    return out


def subtract(a, b):
    """Parts of a not covered by b (boundaries of b are removed as open cuts)."""
    result = []
    for start, end in merge(a):
        pieces = [(start, end)]
        for b_start, b_end in merge(b):
            next_pieces = []
            for p_start, p_end in pieces:
                if b_end <= p_start or b_start >= p_end:
                    next_pieces.append((p_start, p_end))
                    continue
                if b_start > p_start:
                    next_pieces.append((p_start, b_start))
                if b_end < p_end:
                    next_pieces.append((b_end, p_end))
            pieces = next_pieces
        result.extend([list(p) for p in pieces])
    return result


def max_overlap(intervals):
    """Largest number of intervals covering a single point (touching ends count)."""
    events = []
    for iv in intervals:
        start, end = _ordered(iv)
        events.append((start, 0))
        events.append((end, 1))
    depth = best = 0
    for _, kind in sorted(events):
        if kind == 0:
            depth += 1
            best = max(best, depth)
        else:
            depth -= 1
    return best


def describe(intervals):
    parts = [f"[{s}, {e}]" for s, e in merge(intervals)]
    return " U ".join(parts) if parts else "(empty)"
