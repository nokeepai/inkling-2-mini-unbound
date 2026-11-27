import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import *
a = from_dense([0, 3, 0, -2, 0])
b = SparseVector(5, {1: 1, 4: 5})
assert a.nnz() == 2, "nnz"
assert a.density() == 0.4, "density"
assert a.to_dense() == [0, 3, 0, -2, 0], "round trip"
assert a[0] == 0, "implicit zero"
assert add(a, b).to_dense() == [0, 4, 0, -2, 5], "add"
assert add(a, scale(a, -1)).nnz() == 0, "cancellation drops entries"
assert dot(a, b) == 3, "dot"
assert dot(a, a) == 13, "self dot"
assert norm1(a) == 5, "l1 norm"
assert argmax(a) == 1, "argmax"
assert argmax(scale(a, -1)) == 3, "argmax after negation"
assert argmax(SparseVector(3)) == 0, "argmax of zero vector"
assert slice_vector(a, 1, 4).to_dense() == [3, 0, -2], "slice"
assert slice_vector(a, 4, 5).to_dense() == [0], "slice at end"
assert top_k(b, 1) == [4], "top one"
assert top_k(a, 5) == [1, 3], "top k clamps"
a[1] = 0
assert a.nnz() == 1, "setting zero removes"
assert SparseVector(3, {0: 0}).nnz() == 0, "zero entries skipped"
