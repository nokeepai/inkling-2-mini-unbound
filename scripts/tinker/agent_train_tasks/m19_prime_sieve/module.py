"""Prime sieve utilities: sieve, factorization, gaps, twins and Goldbach pairs."""


def sieve(limit):
    """Boolean table where table[i] tells whether i is prime, for 0..limit."""
    if limit < 0:
        return []
    table = [True] * (limit + 1)
    table[0] = False
    if limit >= 1:
        table[1] = False
    p = 2
    while p * p <= limit:
        if table[p]:
            for multiple in range(p * p, limit + 1, p):
                table[multiple] = False
        p += 1
    return table


def primes_up_to(limit):
    table = sieve(limit)
    return [i for i in range(len(table)) if table[i]]


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def prime_count(limit):
    return len(primes_up_to(limit))


def nth_prime(n):
    if n < 1:
        raise ValueError("n must be at least 1")
    count = 0
    candidate = 1
    while count < n:
        candidate += 1
        if is_prime(candidate):
            count += 1
    return candidate


def factorize(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def twin_primes(limit):
    ps = primes_up_to(limit)
    return [(ps[i], ps[i + 1]) for i in range(len(ps) - 1) if ps[i + 1] - ps[i] == 2]


def largest_gap(limit):
    """(gap, lower, upper) for the first largest gap between consecutive primes."""
    ps = primes_up_to(limit)
    best = (0, 0, 0)
    for i in range(1, len(ps)):
        gap = ps[i] - ps[i - 1]
        if gap > best[0]:
            best = (gap, ps[i - 1], ps[i])
    return best


def goldbach(n):
    """Prime pair with the smallest first member summing to even n > 2, or None."""
    if n <= 2 or n % 2 == 1:
        return None
    for p in range(2, n // 2 + 1):
        if is_prime(p) and is_prime(n - p):
            return (p, n - p)
    return None
