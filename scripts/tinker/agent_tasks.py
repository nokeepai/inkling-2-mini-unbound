"""20 harmless buggy Python functions for agent_eval.py: (name, buggy code, tests, reference fix).

Run directly to check every buggy version fails and every reference passes.
"""
from __future__ import annotations

TASKS: list[dict[str, str]] = [
    {"name": "mean", "buggy": "def mean(xs):\n    return sum(xs) / len(xs) - 1\n",
     "fix": "def mean(xs):\n    return sum(xs) / len(xs)\n",
     "tests": "assert mean([1, 2, 3]) == 2\nassert mean([5]) == 5\nassert mean([2, 4]) == 3"},
    {"name": "is_palindrome", "buggy": "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1][1:]\n",
     "fix": "def is_palindrome(s):\n    s = ''.join(c for c in s.lower() if c.isalnum())\n    return s == s[::-1]\n",
     "tests": "assert is_palindrome('Racecar')\nassert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('hello')"},
    {"name": "fizzbuzz", "buggy": "def fizzbuzz(n):\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    return str(n)\n",
     "fix": "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)\n",
     "tests": "assert fizzbuzz(15) == 'FizzBuzz'\nassert fizzbuzz(9) == 'Fizz'\nassert fizzbuzz(10) == 'Buzz'\nassert fizzbuzz(7) == '7'"},
    {"name": "binary_search", "buggy": "def binary_search(xs, t):\n    lo, hi = 0, len(xs)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if xs[mid] < t:\n            lo = mid\n        else:\n            hi = mid\n    return lo if lo < len(xs) and xs[lo] == t else -1\n",
     "fix": "def binary_search(xs, t):\n    lo, hi = 0, len(xs)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if xs[mid] < t:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo if lo < len(xs) and xs[lo] == t else -1\n",
     "tests": "assert binary_search([1, 3, 5, 7], 5) == 2\nassert binary_search([1, 3, 5, 7], 4) == -1\nassert binary_search([1, 3, 5, 7], 7) == 3\nassert binary_search([], 1) == -1"},
    {"name": "word_count", "buggy": "def word_count(text):\n    counts = {}\n    for w in text.split(' '):\n        counts[w] = counts.get(w, 0) + 1\n    return counts\n",
     "fix": "def word_count(text):\n    counts = {}\n    for w in text.lower().split():\n        counts[w] = counts.get(w, 0) + 1\n    return counts\n",
     "tests": "assert word_count('a b  a') == {'a': 2, 'b': 1}\nassert word_count('Hi hi') == {'hi': 2}\nassert word_count('') == {}"},
    {"name": "flatten", "buggy": "def flatten(xs):\n    out = []\n    for x in xs:\n        if isinstance(x, list):\n            out.append(flatten(x))\n        else:\n            out.append(x)\n    return out\n",
     "fix": "def flatten(xs):\n    out = []\n    for x in xs:\n        if isinstance(x, list):\n            out.extend(flatten(x))\n        else:\n            out.append(x)\n    return out\n",
     "tests": "assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]\nassert flatten([]) == []\nassert flatten([[[]]]) == []"},
    {"name": "factorial", "buggy": "def factorial(n):\n    result = 0\n    for i in range(1, n):\n        result *= i\n    return result\n",
     "fix": "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n",
     "tests": "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120"},
    {"name": "chunk", "buggy": "def chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs) - n, n)]\n",
     "fix": "def chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n",
     "tests": "assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert chunk([1, 2], 2) == [[1, 2]]\nassert chunk([], 3) == []"},
    {"name": "second_largest", "buggy": "def second_largest(xs):\n    s = sorted(xs)\n    return s[-2]\n",
     "fix": "def second_largest(xs):\n    s = sorted(set(xs))\n    return s[-2] if len(s) > 1 else None\n",
     "tests": "assert second_largest([3, 1, 4, 4]) == 3\nassert second_largest([5, 5]) is None\nassert second_largest([1, 2]) == 1"},
    {"name": "caesar", "buggy": "def caesar(s, k):\n    out = ''\n    for c in s:\n        if c.isalpha():\n            out += chr(ord(c) + k)\n        else:\n            out += c\n    return out\n",
     "fix": "def caesar(s, k):\n    out = ''\n    for c in s:\n        if c.isalpha():\n            base = ord('A') if c.isupper() else ord('a')\n            out += chr((ord(c) - base + k) % 26 + base)\n        else:\n            out += c\n    return out\n",
     "tests": "assert caesar('abc', 1) == 'bcd'\nassert caesar('xyz', 3) == 'abc'\nassert caesar('Hello, World!', 13) == 'Uryyb, Jbeyq!'"},
    {"name": "merge_sorted", "buggy": "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out\n",
     "fix": "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n",
     "tests": "assert merge_sorted([1, 4], [2, 3, 5]) == [1, 2, 3, 4, 5]\nassert merge_sorted([], [1]) == [1]\nassert merge_sorted([2], []) == [2]"},
    {"name": "count_vowels", "buggy": "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",
     "fix": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
     "tests": "assert count_vowels('Apple') == 2\nassert count_vowels('xyz') == 0\nassert count_vowels('AEIOU') == 5"},
    {"name": "running_max", "buggy": "def running_max(xs):\n    out, best = [], 0\n    for x in xs:\n        best = max(best, x)\n        out.append(best)\n    return out\n",
     "fix": "def running_max(xs):\n    out, best = [], float('-inf')\n    for x in xs:\n        best = max(best, x)\n        out.append(best)\n    return out\n",
     "tests": "assert running_max([1, 3, 2]) == [1, 3, 3]\nassert running_max([-5, -2, -9]) == [-5, -2, -2]\nassert running_max([]) == []"},
    {"name": "dedupe", "buggy": "def dedupe(xs):\n    return list(set(xs))\n",
     "fix": "def dedupe(xs):\n    seen, out = set(), []\n    for x in xs:\n        if x not in seen:\n            seen.add(x); out.append(x)\n    return out\n",
     "tests": "assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]\nassert dedupe([]) == []\nassert dedupe(['b', 'a', 'b']) == ['b', 'a']"},
    {"name": "parse_duration", "buggy": "def parse_duration(s):\n    m, sec = s.split(':')\n    return int(m) * 100 + int(sec)\n",
     "fix": "def parse_duration(s):\n    parts = [int(p) for p in s.split(':')]\n    total = 0\n    for p in parts:\n        total = total * 60 + p\n    return total\n",
     "tests": "assert parse_duration('1:30') == 90\nassert parse_duration('0:05') == 5\nassert parse_duration('1:00:00') == 3600"},
    {"name": "transpose", "buggy": "def transpose(m):\n    return [[m[i][j] for i in range(len(m[0]))] for j in range(len(m))]\n",
     "fix": "def transpose(m):\n    if not m:\n        return []\n    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]\n",
     "tests": "assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]\nassert transpose([[1]]) == [[1]]\nassert transpose([]) == []"},
    {"name": "gcd", "buggy": "def gcd(a, b):\n    while b:\n        a, b = b, a // b\n    return a\n",
     "fix": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)\n",
     "tests": "assert gcd(12, 18) == 6\nassert gcd(7, 5) == 1\nassert gcd(0, 9) == 9"},
    {"name": "title_case", "buggy": "def title_case(s):\n    return ' '.join(w[0].upper() + w[1:] for w in s.split(' '))\n",
     "fix": "def title_case(s):\n    return ' '.join(w[:1].upper() + w[1:].lower() for w in s.split())\n",
     "tests": "assert title_case('hello world') == 'Hello World'\nassert title_case('hELLO  there') == 'Hello There'\nassert title_case('') == ''"},
    {"name": "rotate", "buggy": "def rotate(xs, k):\n    return xs[k:] + xs[:k]\n",
     "fix": "def rotate(xs, k):\n    if not xs:\n        return []\n    k %= len(xs)\n    return xs[-k:] + xs[:-k] if k else xs[:]\n",
     "tests": "assert rotate([1, 2, 3, 4], 1) == [4, 1, 2, 3]\nassert rotate([1, 2, 3], 4) == [3, 1, 2]\nassert rotate([], 2) == []"},
    {"name": "valid_brackets", "buggy": "def valid_brackets(s):\n    depth = 0\n    for c in s:\n        if c in '([{':\n            depth += 1\n        elif c in ')]}':\n            depth -= 1\n    return depth == 0\n",
     "fix": "def valid_brackets(s):\n    pairs, stack = {')': '(', ']': '[', '}': '{'}, []\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n                return False\n    return not stack\n",
     "tests": "assert valid_brackets('([]{})')\nassert not valid_brackets('(]')\nassert not valid_brackets(')(')\nassert valid_brackets('')"},
]


def _check() -> None:
    import code_eval as ce

    for t in TASKS:
        bad, _ = ce.run_tests(f"```python\n{t['buggy']}```", t["tests"].splitlines())
        good, err = ce.run_tests(f"```python\n{t['fix']}```", t["tests"].splitlines())
        assert not bad and good, (t["name"], bad, good, err)
    print(f"{len(TASKS)} tasks ok: buggy versions fail, fixes pass")


if __name__ == "__main__":
    _check()
