import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

g = from_edges([("a", "b", 4), ("a", "c", 1), ("c", "b", 2), ("b", "d", 1), ("c", "d", 5), ("d", "e", 3)])
assert g.shortest_path("a", "d") == (["a", "c", "b", "d"], 4), "a to d"
assert g.shortest_path("e", "a") == (["e", "d", "b", "c", "a"], 7), "e to a"
g.add_edge("a", "e", 6)
g.add_edge("e", "a", 9)
assert g.shortest_path("a", "e") == (["a", "e"], 6), "a to e after adding a-e 6 and e-a 9"
assert g.shortest_path("e", "a") == (["e", "a"], 6), "e to a after adding a-e 6 and e-a 9"
assert g.edge_count() == 7 and g.eccentricity("a") == 6, "edge count and eccentricity"
dag = from_edges([("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")], directed=True)
assert dag.has_cycle() is False, "s->x->t, s->y->t"
assert dag.shortest_path("t", "s") == (None, float("inf")), "t to s in the dag"
dag.add_edge("t", "s")
assert dag.has_cycle() is True, "after adding t->s"
assert sorted(dag.reachable("x")) == ["s", "t", "x", "y"], "reachable from x"
g2 = Graph()
g2.add_edge(1, 2)
g2.add_edge(3, 4)
g2.add_node(5)
assert g2.components() == [[1, 2], [3, 4], [5]], "components"
