"""Small dense-matrix helpers using exact Fraction arithmetic."""
from fractions import Fraction


def zeros(rows, cols):
    return [[Fraction(0)] * cols] * rows


def identity(n):
    m = zeros(n, n)
    for i in range(n):
        m[i][i] = Fraction(1)
    return m


def shape(m):
    return (len(m), len(m[0]) if m else 0)


def to_fractions(m):
    return [[Fraction(x) for x in row] for row in m]


def transpose(m):
    rows, cols = shape(m)
    out = zeros(cols, rows)
    for i in range(rows):
        for j in range(cols):
            out[j][i] = m[i][j]
    return out


def add(a, b):
    if shape(a) != shape(b):
        raise ValueError("shape mismatch")
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def scale(m, k):
    return [[x * k for x in row] for row in m]


def matmul(a, b):
    ar, ac = shape(a)
    br, bc = shape(b)
    if ac != br:
        raise ValueError("shape mismatch")
    out = zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            out[i][j] = sum((a[i][k] * b[k][j] for k in range(ac)), Fraction(0))
    return out


def power(m, k):
    """m raised to a non-negative integer power by repeated squaring."""
    n, cols = shape(m)
    if n != cols:
        raise ValueError("square matrix required")
    result = identity(n)
    base = to_fractions(m)
    while k > 0:
        if k & 1:
            result = matmul(result, base)
        base = matmul(base, base)
        k >>= 1
    return result


def trace(m):
    return sum((m[i][i] for i in range(len(m))), Fraction(0))


def determinant(m):
    n, cols = shape(m)
    if n != cols:
        raise ValueError("square matrix required")
    a = to_fractions(m)
    det = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        det *= a[col][col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for c in range(col, n):
                a[r][c] -= factor * a[col][c]
    return det


def inverse(m):
    """Gauss-Jordan inverse; raises ValueError for singular matrices."""
    n, cols = shape(m)
    if n != cols:
        raise ValueError("square matrix required")
    a = to_fractions(m)
    inv = identity(n)
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
        if pivot is None:
            raise ValueError("matrix is singular")
        a[col], a[pivot] = a[pivot], a[col]
        inv[col], inv[pivot] = inv[pivot], inv[col]
        lead = a[col][col]
        a[col] = [x / lead for x in a[col]]
        inv[col] = [x / lead for x in inv[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                factor = a[r][col]
                a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
                inv[r] = [x - factor * y for x, y in zip(inv[r], inv[col])]
    return inv


def solve(a, b):
    """Solve a x = b for a vector b, returned as a list."""
    column = [[Fraction(v)] for v in b]
    return [row[0] for row in matmul(inverse(a), column)]


def format_matrix(m):
    cells = [[str(x) for x in row] for row in m]
    width = max((len(c) for row in cells for c in row), default=0)
    return "\n".join(" ".join(c.rjust(width) for c in row) for row in cells)
