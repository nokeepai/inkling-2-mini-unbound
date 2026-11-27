"""Fixed-size bitset backed by a single Python integer."""


class BitSet:
    def __init__(self, size):
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self.bits = 0

    def _check(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("bit out of range")

    def set(self, i):
        self._check(i)
        self.bits |= 1 << i

    def clear(self, i):
        self._check(i)
        self.bits &= ~(1 << i)

    def toggle(self, i):
        self._check(i)
        self.bits ^= 1 << i

    def test(self, i):
        self._check(i)
        return (self.bits >> i) & 1 == 1

    def count(self):
        n = 0
        for i in range(self.size):
            if (self.bits >> i) & 1:
                n += 1
        return n

    def members(self):
        return [i for i in range(self.size) if self.test(i)]

    def set_range(self, start, stop):
        for i in range(start, stop):
            self.set(i)

    def first_clear(self):
        for i in range(self.size):
            if not self.test(i):
                return i
        return -1

    def to_string(self):
        """Most significant bit first."""
        return "".join("1" if self.test(i) else "0" for i in range(self.size - 1, -1, -1))

    def longest_run(self):
        best = 0
        run = 0
        for i in range(self.size):
            if self.test(i):
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best


def from_members(size, members):
    b = BitSet(size)
    for m in members:
        b.set(m)
    return b


def union(a, b):
    out = BitSet(max(a.size, b.size))
    out.bits = a.bits | b.bits
    return out


def intersection(a, b):
    out = BitSet(min(a.size, b.size))
    out.bits = a.bits & b.bits & ((1 << out.size) - 1)
    return out


def difference(a, b):
    out = BitSet(a.size)
    out.bits = a.bits & ~b.bits
    return out
