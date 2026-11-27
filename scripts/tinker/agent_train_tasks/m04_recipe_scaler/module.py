"""Recipe scaler with exact fractions, unit normalization and friendly display."""

TSP_PER_UNIT = {"tsp": 1, "tbsp": 3, "cup": 48}


def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def simplify(num, den):
    if den == 0:
        raise ValueError("zero denominator")
    if den < 0:
        num, den = -num, -den
    g = gcd(num, den)
    if g == 0:
        return (0, 1)
    return (num // g, den // g)


def scale(num, den, servings_from, servings_to):
    return simplify(num * servings_to, den * servings_from)


def format_fraction(num, den):
    num, den = simplify(num, den)
    whole = num // den
    rem = num - whole * den
    if rem == 0:
        return str(whole)
    if whole == 0:
        return f"{rem}/{den}"
    return f"{whole} {rem}/{den}"


def to_tsp(num, den, unit):
    return simplify(num * TSP_PER_UNIT[unit], den)


def best_unit(num_tsp, den_tsp):
    """Pick the largest unit in which the amount is at least 1."""
    for unit in ("cup", "tbsp"):
        factor = TSP_PER_UNIT[unit]
        if num_tsp >= factor * den_tsp:
            return unit, simplify(num_tsp, den_tsp * factor)
    return "tsp", simplify(num_tsp, den_tsp)


class Recipe:
    def __init__(self, servings):
        self.servings = servings
        self.ingredients = []

    def add(self, name, num, den, unit):
        self.ingredients.append((name, num, den, unit))

    def scaled(self, new_servings):
        out = []
        for name, num, den, unit in self.ingredients:
            n, d = scale(num, den, self.servings, new_servings)
            tn, td = to_tsp(n, d, unit)
            best, (bn, bd) = best_unit(tn, td)
            out.append((name, format_fraction(bn, bd), best))
        return out

    def total_tsp(self):
        tn, td = 0, 1
        for _, num, den, unit in self.ingredients:
            n, d = to_tsp(num, den, unit)
            tn, td = simplify(tn * d + n * td, td * d)
        return (tn, td)


def servings_for(available_num, available_den, need_num, need_den, base_servings):
    """Whole servings possible given an available amount and per-recipe need."""
    num = available_num * need_den * base_servings
    den = available_den * need_num
    return num // den
