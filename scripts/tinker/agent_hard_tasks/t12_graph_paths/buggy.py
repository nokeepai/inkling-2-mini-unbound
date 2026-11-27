"""Weighted graph: Dijkstra shortest paths, reachability, components and cycle check."""
import heapq
from collections import deque

INF = float("inf")


class Graph:
    def __init__(self, directed=False):
        self.directed = directed
        self.adj = {}

    def add_node(self, node):
        self.adj.setdefault(node, {})

    def add_edge(self, u, v, weight=1):
        """Add an edge; adding an edge that already exists keeps the smaller weight."""
        if weight < 0:
            raise ValueError("negative weights are not supported")
        self.add_node(u)
        self.add_node(v)
        self.adj[u][v] = min(weight, self.adj[u].get(v, INF))
        if not self.directed:
            self.adj[v][u] = weight

    def nodes(self):
        return sorted(self.adj)

    def edge_count(self):
        total = sum(len(nbrs) for nbrs in self.adj.values())
        return total if self.directed else total // 2

    def dijkstra(self, source):
        """Shortest distances from source and the predecessor map."""
        dist = {source: 0}
        prev = {}
        heap = [(0, source)]
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for v, w in sorted(self.adj.get(u, {}).items()):
                nd = d + w
                if v not in dist:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return dist, prev

    def shortest_path(self, source, target):
        """(path, cost) of a cheapest route, or (None, inf) when unreachable."""
        dist, prev = self.dijkstra(source)
        if target not in dist:
            return None, INF
        path = [target]
        while path[-1] != source:
            path.append(prev[path[-1]])
        path.reverse()
        return path, dist[target]

    def path_weight(self, path):
        return sum(self.adj[a][b] for a, b in zip(path, path[1:]))

    def reachable(self, source):
        seen = {source}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in self.adj.get(u, {}):
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        return seen

    def components(self):
        """Connected components of the undirected view, each sorted, ordered by first member."""
        undirected = {n: set(nbrs) for n, nbrs in self.adj.items()}
        for u, nbrs in self.adj.items():
            for v in nbrs:
                undirected[v].add(u)
        seen, comps = set(), []
        for start in sorted(self.adj):
            if start in seen:
                continue
            comp, stack = [], [start]
            seen.add(start)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in undirected[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(sorted(comp))
        return comps

    def has_cycle(self):
        """True if the directed graph contains a cycle."""
        if not self.directed:
            raise ValueError("cycle check is for directed graphs")
        white, grey, black = 0, 1, 2
        color = {n: white for n in self.adj}

        def visit(u):
            color[u] = grey
            for v in self.adj[u]:
                if color[v] != white:
                    return True
                if color[v] == white and visit(v):
                    return True
            color[u] = black
            return False

        return any(color[n] == white and visit(n) for n in sorted(self.adj))

    def eccentricity(self, node):
        dist, _ = self.dijkstra(node)
        if len(dist) < len(self.adj):
            return INF
        return max(dist.values())


def from_edges(edges, directed=False):
    g = Graph(directed)
    for edge in edges:
        g.add_edge(*edge)
    return g
