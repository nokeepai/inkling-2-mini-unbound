"""Deterministic single-car elevator simulator using the SCAN strategy."""


class Elevator:
    def __init__(self, floors, start=0):
        if start < 0 or start >= floors:
            raise ValueError("start floor out of range")
        self.floors = floors
        self.floor = start
        self.direction = 0
        self.requests = set()
        self.visited = []
        self.travelled = 0

    def call(self, floor):
        """Register a stop; calls for the current floor are served immediately."""
        if floor < 0 or floor >= self.floors:
            return False
        if floor == self.floor:
            return False
        self.requests.add(floor)
        return True

    def _pending_above(self):
        return any(f > self.floor for f in self.requests)

    def _pending_below(self):
        return any(f < self.floor for f in self.requests)

    def choose_direction(self):
        """Keep going the same way while requests remain ahead, else reverse."""
        if not self.requests:
            return 0
        if self.direction >= 0 and self._pending_above():
            return 1
        if self.direction <= 0 and self._pending_below():
            return -1
        return 1 if self._pending_above() else -1

    def step(self):
        """Move one floor; stop and record if that floor was requested."""
        self.direction = self.choose_direction()
        if self.direction == 0:
            return False
        self.floor += self.direction
        self.travelled += 1
        if self.floor in self.requests:
            self.requests.discard(self.floor)
            self.visited.append(self.floor)
        if not self.requests:
            self.direction = 0
        return True

    def run(self, max_steps=1000):
        steps = 0
        while self.requests and steps < max_steps:
            self.step()
            steps += 1
        return steps


def total_distance(start, stops):
    total = 0
    pos = start
    for s in stops:
        total += abs(s - pos)
        pos = s
    return total


def simulate(floors, start, calls):
    e = Elevator(floors, start)
    for f in calls:
        e.call(f)
    e.run()
    return e.visited, e.travelled


def fifo_distance(start, calls):
    """Distance if calls were served strictly in arrival order (no SCAN)."""
    stops = []
    for f in calls:
        if not stops or stops[-1] != f:
            stops.append(f)
    return total_distance(start, stops)
