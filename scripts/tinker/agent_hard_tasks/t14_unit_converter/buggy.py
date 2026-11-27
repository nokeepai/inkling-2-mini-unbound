"""Unit converter: a graph of linear/affine conversions with chained lookups and a cache."""
from collections import deque


class Conversion:
    """Maps x to scale * x + offset."""

    def __init__(self, scale, offset=0.0):
        if scale == 0:
            raise ValueError("scale must be non-zero")
        self.scale = scale
        self.offset = offset

    def apply(self, x):
        return self.scale * x + self.offset

    def inverse(self):
        return Conversion(1 / self.scale, -self.offset)

    def then(self, other):
        """The conversion that applies self first and other second."""
        return Conversion(self.scale * other.scale, self.scale * other.offset + self.offset)

    def __repr__(self):
        return f"Conversion({self.scale!r}, {self.offset!r})"


IDENTITY = Conversion(1.0)


class UnitConverter:
    def __init__(self):
        self.edges = {}
        self._cache = {}

    def define(self, src, dst, scale, offset=0.0):
        """Declare that a value v in src equals scale * v + offset in dst."""
        conv = Conversion(scale, offset)
        self.edges.setdefault(src, {})[dst] = conv
        self.edges.setdefault(dst, {})[src] = conv.inverse()
        self._cache.clear()

    def units(self):
        return sorted(self.edges)

    def _find(self, src, dst):
        if src not in self.edges or dst not in self.edges:
            raise KeyError(f"unknown unit {src if src not in self.edges else dst}")
        if src == dst:
            return IDENTITY
        seen = {src}
        queue = deque([(src, IDENTITY)])
        while queue:
            unit, conv = queue.popleft()
            for nxt in sorted(self.edges[unit]):
                if nxt in seen:
                    continue
                step = conv.then(self.edges[unit][nxt])
                if nxt == dst:
                    return step
                seen.add(nxt)
                queue.append((nxt, step))
        raise KeyError(f"no conversion from {src} to {dst}")

    def conversion(self, src, dst):
        key = (src, dst)
        if key not in self._cache:
            conv = self._find(src, dst)
            self._cache[key] = conv
            self._cache[(dst, src)] = conv
        return self._cache[key]

    def convert(self, value, src, dst, digits=6):
        return round(self.conversion(src, dst).apply(value), digits)

    def convert_many(self, values, src, dst, digits=6):
        conv = self.conversion(src, dst)
        return [round(conv.apply(v), digits) for v in values]

    def reachable_units(self, src):
        seen, queue = {src}, deque([src])
        while queue:
            unit = queue.popleft()
            for nxt in self.edges.get(unit, {}):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return sorted(seen - {src})

    def best_unit(self, value, src, candidates):
        """Candidate giving the smallest converted magnitude that is still at least 1."""
        best = None
        for unit in candidates:
            converted = abs(self.convert(value, src, unit))
            if converted >= 1 and (best is None or converted < best[0]):
                best = (converted, unit)
        return None if best is None else best[1]


def standard_converter():
    uc = UnitConverter()
    uc.define("km", "m", 1000)
    uc.define("m", "cm", 100)
    uc.define("mile", "km", 1.609344)
    uc.define("ft", "in", 12)
    uc.define("in", "cm", 2.54)
    uc.define("C", "F", 1.8, 32)
    uc.define("C", "K", 1, 273.15)
    uc.define("K", "R", 1.8)
    return uc


def parse_quantity(text):
    """'12.5 km' -> (12.5, 'km')."""
    number, unit = text.split()
    return float(number), unit
