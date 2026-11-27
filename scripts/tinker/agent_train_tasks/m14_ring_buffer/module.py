"""Fixed-capacity ring buffer that overwrites the oldest item when full."""


class RingBuffer:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.data = [None] * capacity
        self.head = 0
        self.length = 0
        self.dropped = 0

    def push(self, item):
        tail = (self.head + self.length) % self.capacity
        self.data[tail] = item
        if self.length < self.capacity:
            self.length += 1
        else:
            # buffer was full: the write replaced the oldest item
            self.head = (self.head + 1) % self.capacity
            self.dropped += 1

    def pop(self):
        """Remove and return the oldest item."""
        if self.length == 0:
            raise IndexError("empty buffer")
        item = self.data[self.head]
        self.data[self.head] = None
        self.head = (self.head + 1) % self.capacity
        self.length -= 1
        return item

    def peek(self, offset=0):
        if offset < 0 or offset >= self.length:
            raise IndexError("offset out of range")
        return self.data[(self.head + offset) % self.capacity]

    def newest(self):
        return self.peek(self.length - 1)

    def is_full(self):
        return self.length == self.capacity

    def to_list(self):
        return [self.data[(self.head + i) % self.capacity] for i in range(self.length)]

    def extend(self, items):
        for item in items:
            self.push(item)

    def last_n(self, n):
        n = min(n, self.length)
        return self.to_list()[self.length - n:]

    def average(self):
        if self.length == 0:
            return 0.0
        return sum(self.to_list()) / self.length


def rolling_max(values, capacity):
    buf = RingBuffer(capacity)
    out = []
    for v in values:
        buf.push(v)
        out.append(max(buf.to_list()))
    return out


def rolling_sum_exceeds(values, capacity, limit):
    """Indices at which the sum of the last `capacity` values goes above limit."""
    buf = RingBuffer(capacity)
    hits = []
    for i in range(len(values)):
        buf.push(values[i])
        if sum(buf.to_list()) > limit:
            hits.append(i)
    return hits
