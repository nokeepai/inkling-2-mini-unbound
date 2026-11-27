"""Dependency-aware task scheduler driven by a priority queue.

Lower priority numbers run first; tasks with equal priority run in the order they were added.
"""
import heapq


class CycleError(ValueError):
    pass


class Scheduler:
    def __init__(self):
        self.tasks = {}
        self._added = {}
        self._seq = 0

    def add(self, name, priority=0, deps=(), duration=1):
        if name in self.tasks:
            raise ValueError(f"duplicate task {name}")
        self.tasks[name] = {"priority": priority, "deps": set(deps), "duration": duration}
        self._added[name] = self._seq
        self._seq += 1

    def set_priority(self, name, priority):
        self.tasks[name]["priority"] = priority

    def _entry(self, name):
        return (self.tasks[name]["priority"], name)

    def _dependents(self):
        dependents = dict.fromkeys(self.tasks, [])
        for name, task in self.tasks.items():
            for dep in task["deps"]:
                if dep in dependents:
                    dependents[dep].append(name)
        return dependents

    def _check_missing(self):
        missing = {d for t in self.tasks.values() for d in t["deps"] if d not in self.tasks}
        if missing:
            raise KeyError(sorted(missing)[0])

    def _indegrees(self):
        return {name: len(t["deps"]) for name, t in self.tasks.items()}

    def order(self):
        """Execution order: always the most urgent task whose dependencies are done."""
        self._check_missing()
        indegree = self._indegrees()
        dependents = self._dependents()
        heap = [self._entry(n) for n, deg in indegree.items() if deg == 0]
        heapq.heapify(heap)
        result = []
        while heap:
            name = heapq.heappop(heap)[-1]
            result.append(name)
            for nxt in dependents[name]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(heap, self._entry(nxt))
        if len(result) != len(self.tasks):
            stuck = sorted(n for n in self.tasks if n not in result)
            raise CycleError("cycle among " + ", ".join(stuck))
        return result

    def ready(self, done):
        """Tasks not yet done whose dependencies are all in `done`, most urgent first."""
        done = set(done)
        names = [n for n, t in self.tasks.items() if n not in done and t["deps"] <= done]
        return [entry[-1] for entry in sorted(self._entry(n) for n in names)]

    def cancel(self, name):
        """Remove a task and everything that depends on it, directly or indirectly."""
        dependents = self._dependents()
        removed, stack = set(), [name]
        while stack:
            current = stack.pop()
            for child in dependents[current]:
                if child not in removed:
                    removed.add(child)
        removed.add(name)
        for gone in removed:
            del self.tasks[gone]
            del self._added[gone]
        return sorted(removed)

    def makespan(self, workers=1):
        """Finish time when each free worker immediately takes the most urgent ready task."""
        self.order()
        indegree = self._indegrees()
        dependents = self._dependents()
        ready = [self._entry(n) for n, deg in indegree.items() if deg == 0]
        heapq.heapify(ready)
        running = []
        clock = 0
        while ready or running:
            while ready and len(running) < workers:
                name = heapq.heappop(ready)[-1]
                finish = clock + self.tasks[name]["duration"]
                heapq.heappush(running, (finish, self._added[name], name))
            clock, _, name = heapq.heappop(running)
            for nxt in dependents[name]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(ready, self._entry(nxt))
        return clock

    def critical_path(self):
        """Longest total duration along any dependency chain."""
        finish = {}
        for name in self.order():
            task = self.tasks[name]
            start = max((finish[d] for d in task["deps"]), default=0)
            finish[name] = start + task["duration"]
        return max(finish.values(), default=0)


def from_spec(lines):
    """Build a scheduler from lines like 'name priority duration dep1,dep2'."""
    sched = Scheduler()
    for line in lines:
        parts = line.split()
        deps = parts[3].split(",") if len(parts) > 3 else []
        sched.add(parts[0], int(parts[1]), deps, int(parts[2]))
    return sched
