"""Least-recently-used cache with hit/miss statistics."""


class CacheStats:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def hit_rate(self):
        lookups = self.hits + self.misses
        return 0.0 if lookups == 0 else self.hits / lookups

    def as_dict(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate(), 3),
        }

    def reset(self):
        self.hits = self.misses = self.evictions = 0


class LRUCache:
    """Keys are kept in recency order: self._order[0] is the least recently used."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._data = {}
        self._order = []
        self.stats = CacheStats()
        self.evicted = []

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    def _touch(self, key):
        if key not in self._order:
            self._order.append(key)

    def _evict_one(self):
        oldest = self._order.pop(0)
        del self._data[oldest]
        self.stats.evictions += 1
        self.evicted.append(oldest)

    def get(self, key, default=None):
        if key in self._data:
            self.stats.hits += 1
            self._touch(key)
            return self._data[key]
        self.stats.misses += 1
        return default

    def peek(self, key, default=None):
        return self._data.get(key, default)

    def put(self, key, value):
        if len(self._data) >= self.capacity:
            self._evict_one()
        self._data[key] = value
        self._touch(key)

    def pop(self, key, default=None):
        if key not in self._data:
            return default
        return self._data.pop(key)

    def keys(self):
        return list(self._order)

    def items(self):
        return [(k, self._data[k]) for k in self._order]

    def resize(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        while len(self._data) > self.capacity:
            self._evict_one()

    def clear(self):
        self._data.clear()
        self._order.clear()

    def get_or_compute(self, key, compute):
        if key in self._data:
            return self.get(key)
        self.stats.misses += 1
        value = compute(key)
        self.put(key, value)
        return value


def memoize(capacity):
    """Decorator caching a one-argument function in an LRUCache exposed as .cache."""
    def wrap(fn):
        cache = LRUCache(capacity)

        def inner(arg):
            return cache.get_or_compute(arg, fn)

        inner.cache = cache
        return inner
    return wrap
