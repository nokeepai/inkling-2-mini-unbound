"""Sliding-window statistics over numeric sequences."""


def window_sums(values, k):
    if k <= 0 or k > len(values):
        return []
    current = sum(values[:k])
    out = [current]
    for i in range(k, len(values)):
        current += values[i] - values[i - k]
        out.append(current)
    return out


def window_means(values, k):
    return [s / k for s in window_sums(values, k)]


def window_max(values, k):
    """Maximum of each window using a monotonic queue of indices."""
    if k <= 0 or k > len(values):
        return []
    dq = []
    out = []
    for i in range(len(values)):
        while dq and dq[0] <= i - k:
            dq.pop(0)
        while dq and values[dq[-1]] <= values[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            out.append(values[dq[0]])
    return out


def window_min(values, k):
    return [-m for m in window_max([-v for v in values], k)]


def best_window(values, k):
    """Start index of the window with the largest sum; earliest wins ties."""
    sums = window_sums(values, k)
    if not sums:
        return -1
    best = 0
    for i in range(1, len(sums)):
        if sums[i] > sums[best]:
            best = i
    return best


def longest_under(values, limit):
    """Longest contiguous run with sum <= limit (values must be non-negative)."""
    left = 0
    total = 0
    best = 0
    for right in range(len(values)):
        total += values[right]
        while total > limit and left <= right:
            total -= values[left]
            left += 1
        best = max(best, right - left + 1)
    return best


def count_above(values, k, threshold):
    return sum(1 for m in window_means(values, k) if m > threshold)


def window_ranges(values, k):
    highs = window_max(values, k)
    lows = window_min(values, k)
    return [highs[i] - lows[i] for i in range(len(highs))]


def distinct_counts(values, k):
    """Number of distinct values in each window of size k."""
    if k <= 0 or k > len(values):
        return []
    seen = {}
    out = []
    for i in range(len(values)):
        seen[values[i]] = seen.get(values[i], 0) + 1
        if i >= k:
            old = values[i - k]
            seen[old] -= 1
            if seen[old] == 0:
                del seen[old]
        if i >= k - 1:
            out.append(len(seen))
    return out
