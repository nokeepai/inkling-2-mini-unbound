"""Polynomial arithmetic on integer coefficient lists, lowest degree first."""


def trim(coeffs):
    out = list(coeffs)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out if out else [0]


def degree(p):
    """Degree of p; the zero polynomial has degree -1."""
    p = trim(p)
    if p == [0]:
        return -1
    return len(p) - 1


def add(p, q):
    n = max(len(p), len(q))
    out = []
    for i in range(n):
        a = p[i] if i < len(p) else 0
        b = q[i] if i < len(q) else 0
        out.append(a + b)
    return trim(out)


def sub(p, q):
    return add(p, [-c for c in q])


def mul(p, q):
    out = [0] * (len(p) + len(q) - 1)
    for i in range(len(p)):
        for j in range(len(q)):
            out[i + j] += p[i] * q[j]
    return trim(out)


def evaluate(p, x):
    """Horner's method."""
    result = 0
    for c in reversed(p):
        result = result * x + c
    return result


def derivative(p):
    if len(p) <= 1:
        return [0]
    return trim([i * p[i] for i in range(1, len(p))])


def compose(p, q):
    """Return p(q(x))."""
    result = [0]
    for c in reversed(p):
        result = add(mul(result, q), [c])
    return result


def divmod_poly(p, d):
    """Long division by a monic divisor; returns (quotient, remainder)."""
    d = trim(d)
    if d[-1] != 1:
        raise ValueError("divisor must be monic")
    rem = trim(p)
    if len(rem) < len(d):
        return [0], rem
    quot = [0] * (len(rem) - len(d) + 1)
    for shift in range(len(quot) - 1, -1, -1):
        coef = rem[shift + len(d) - 1]
        quot[shift] = coef
        for j in range(len(d)):
            rem[shift + j] -= coef * d[j]
    return trim(quot), trim(rem[:len(d) - 1] or [0])


def to_string(p):
    p = trim(p)
    terms = []
    for i in range(len(p) - 1, -1, -1):
        c = p[i]
        if c == 0:
            continue
        body = "" if (abs(c) == 1 and i > 0) else str(abs(c))
        if i >= 2:
            body += f"x^{i}"
        elif i == 1:
            body += "x"
        terms.append(("-" if c < 0 else "+", body))
    if not terms:
        return "0"
    first_sign, first_body = terms[0]
    out = ("-" if first_sign == "-" else "") + first_body
    for sign, body in terms[1:]:
        out += f" {sign} {body}"
    return out
