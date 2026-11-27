"""Roman numeral conversion, validation and arithmetic."""

VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]
SYMBOLS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def to_roman(n):
    if n < 1 or n > 3999:
        raise ValueError("out of range")
    out = []
    for value, sym in VALUES:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def from_roman(text):
    total = 0
    for i in range(len(text)):
        v = SYMBOLS[text[i]]
        if i + 1 < len(text) and SYMBOLS[text[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


def is_valid(text):
    """Canonical numerals only: the text must round-trip exactly."""
    if not text or any(ch not in SYMBOLS for ch in text):
        return False
    n = from_roman(text)
    return 1 <= n and n <= 3999 and to_roman(n) == text


def add(a, b):
    return to_roman(from_roman(a) + from_roman(b))


def subtract(a, b):
    return to_roman(from_roman(a) - from_roman(b))


def longest_numeral(limit):
    """Smallest n in 1..limit whose numeral is longest."""
    best_n, best_len = 1, 1
    for n in range(1, limit + 1):
        length = len(to_roman(n))
        if length > best_len:
            best_n, best_len = n, length
    return best_n


def count_symbols(text):
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    return counts


def max_repeat(text):
    best = 0
    run = 0
    prev = ""
    for ch in text:
        run = run + 1 if ch == prev else 1
        prev = ch
        best = max(best, run)
    return best


def sort_numerals(items):
    return sorted(items, key=from_roman)


def numerals_between(lo, hi):
    return [to_roman(n) for n in range(max(lo, 1), min(hi, 3999) + 1)]
