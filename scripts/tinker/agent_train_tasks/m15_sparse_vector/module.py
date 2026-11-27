"""Sparse vectors stored as {index: value} with zeros omitted."""


class SparseVector:
    def __init__(self, dim, entries=None):
        self.dim = dim
        self.entries = {}
        for i, v in (entries or {}).items():
            self[i] = v

    def __setitem__(self, i, value):
        if i < 0 or i >= self.dim:
            raise IndexError("index out of range")
        if value == 0:
            self.entries.pop(i, None)
        else:
            self.entries[i] = value

    def __getitem__(self, i):
        if i < 0 or i >= self.dim:
            raise IndexError("index out of range")
        return self.entries.get(i, 0)

    def nnz(self):
        return len(self.entries)

    def density(self):
        return self.nnz() / self.dim

    def to_dense(self):
        return [self.entries.get(i, 0) for i in range(self.dim)]


def from_dense(values):
    return SparseVector(len(values), {i: v for i, v in enumerate(values) if v != 0})


def add(a, b):
    if a.dim != b.dim:
        raise ValueError("dimension mismatch")
    out = SparseVector(a.dim, a.entries)
    for i, v in b.entries.items():
        out[i] = out[i] + v
    return out


def scale(a, k):
    return SparseVector(a.dim, {i: v * k for i, v in a.entries.items()})


def dot(a, b):
    if a.dim != b.dim:
        raise ValueError("dimension mismatch")
    # iterate over the sparser operand
    small, large = (a, b) if a.nnz() <= b.nnz() else (b, a)
    total = 0
    for i, v in small.entries.items():
        total += v * large[i]
    return total


def norm1(a):
    return sum(abs(v) for v in a.entries.values())


def argmax(a):
    """Index of the largest value (implicit zeros count); lowest index wins ties."""
    best_i, best_v = 0, a[0]
    for i in range(1, a.dim):
        if a[i] > best_v:
            best_i, best_v = i, a[i]
    return best_i


def slice_vector(a, start, stop):
    out = SparseVector(stop - start)
    for i, v in a.entries.items():
        if start <= i and i < stop:
            out[i - start] = v
    return out


def top_k(a, k):
    """Indices of the k largest stored values, ties by lower index."""
    ranked = sorted(a.entries.items(), key=lambda kv: (-kv[1], kv[0]))
    return [i for i, _ in ranked[:k]]
