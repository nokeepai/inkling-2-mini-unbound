"""Polygon geometry on integer lattice points: area, perimeter, Pick's theorem."""


def shoelace2(points):
    """Twice the signed area (positive for counter-clockwise order)."""
    n = len(points)
    total = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total


def area(points):
    if len(points) < 3:
        return 0.0
    return abs(shoelace2(points)) / 2


def is_clockwise(points):
    return shoelace2(points) < 0


def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def perimeter(points):
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points)):
        total += dist2(points[i], points[(i + 1) % len(points)]) ** 0.5
    return total


def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def boundary_points(points):
    n = len(points)
    total = 0
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        total += gcd(b[0] - a[0], b[1] - a[1])
    return total


def interior_points(points):
    """Pick's theorem: A = I + B/2 - 1, so I = (2A - B + 2) / 2."""
    twice = abs(shoelace2(points))
    b = boundary_points(points)
    return (twice - b + 2) // 2


def bounding_box(points):
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def is_convex(points):
    n = len(points)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        c = cross(points[i], points[(i + 1) % n], points[(i + 2) % n])
        if c != 0:
            current = 1 if c > 0 else -1
            if sign != 0 and current != sign:
                return False
            sign = current
    return sign != 0


def contains(points, p):
    """Ray casting test; points exactly on an edge may go either way."""
    inside = False
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        if (y1 > p[1]) != (y2 > p[1]):
            x_cross = x1 + (p[1] - y1) * (x2 - x1) / (y2 - y1)
            if p[0] < x_cross:
                inside = not inside
    return inside
