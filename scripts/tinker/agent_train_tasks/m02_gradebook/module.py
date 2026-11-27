"""Grade book statistics: averages, letter grades, ranking and curving."""

LETTER_CUTOFFS = [(90, "A"), (80, "B"), (70, "C"), (60, "D")]


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values):
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2


def letter(score):
    for cutoff, grade in LETTER_CUTOFFS:
        if score >= cutoff:
            return grade
    return "F"


def drop_lowest(scores, k=1):
    """Drop the k lowest scores, but never drop everything."""
    if k <= 0 or len(scores) <= k:
        return list(scores)
    return sorted(scores)[k:]


def weighted_average(scores, weights):
    if len(scores) != len(weights):
        raise ValueError("length mismatch")
    total_weight = 0
    acc = 0
    for i in range(len(scores)):
        acc += scores[i] * weights[i]
        total_weight += weights[i]
    if total_weight == 0:
        return 0.0
    return acc / total_weight


def curve(scores, target_max=100):
    """Shift every score so the best one lands on target_max."""
    if not scores:
        return []
    top = max(scores)
    bump = target_max - top
    return [min(target_max, s + bump) for s in scores]


def rank(book):
    """Names ordered by average descending, ties broken alphabetically."""
    pairs = [(-mean(v), name) for name, v in book.items()]
    pairs.sort()
    return [name for _, name in pairs]


def distribution(book):
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for name in book:
        counts[letter(mean(book[name]))] += 1
    return counts


def honor_roll(book, threshold=85, min_count=3):
    result = []
    for name in sorted(book):
        scores = book[name]
        if len(scores) >= min_count and mean(scores) >= threshold:
            result.append(name)
    return result


def most_improved(book):
    """Student with the largest gain from first to last score (needs 2+ scores)."""
    best_name = None
    best_gain = 0
    for name in sorted(book):
        scores = book[name]
        if len(scores) < 2:
            continue
        gain = scores[-1] - scores[0]
        if gain > best_gain:
            best_name, best_gain = name, gain
    return best_name
