"""Playlist with a deterministic shuffle built from strides, rotation and riffles."""


def rotate(items, k):
    if not items:
        return []
    k = k % len(items)
    return items[k:] + items[:k]


def interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def riffle(items):
    """Split in half (first half larger when odd) and interleave the halves."""
    mid = (len(items) + 1) // 2
    return interleave(items[:mid], items[mid:])


def stride_order(n, step):
    """Visit 0, step, 2*step ... modulo n; on a visited slot, advance to the next free one."""
    if n == 0:
        return []
    seen = [False] * n
    order = []
    pos = 0
    while len(order) < n:
        while seen[pos]:
            pos = (pos + 1) % n
        seen[pos] = True
        order.append(pos)
        pos = (pos + step) % n
    return order


class Playlist:
    def __init__(self, songs):
        self.songs = list(songs)
        self.cursor = 0

    def current(self):
        if not self.songs:
            return None
        return self.songs[self.cursor]

    def next(self):
        if not self.songs:
            return None
        self.cursor = (self.cursor + 1) % len(self.songs)
        return self.songs[self.cursor]

    def prev(self):
        if not self.songs:
            return None
        self.cursor = (self.cursor - 1) % len(self.songs)
        return self.songs[self.cursor]

    def shuffle(self, seed):
        order = stride_order(len(self.songs), seed)
        self.songs = rotate([self.songs[i] for i in order], seed)
        self.cursor = 0

    def move(self, src, dst):
        song = self.songs.pop(src)
        self.songs.insert(dst, song)

    def window(self, size):
        """The next `size` songs starting at the cursor, wrapping around."""
        n = len(self.songs)
        return [self.songs[(self.cursor + i) % n] for i in range(min(size, n))]


def no_repeat_artist(songs):
    """True if no two consecutive (artist, title) entries share an artist."""
    for i in range(1, len(songs)):
        if songs[i][0] == songs[i - 1][0]:
            return False
    return True
