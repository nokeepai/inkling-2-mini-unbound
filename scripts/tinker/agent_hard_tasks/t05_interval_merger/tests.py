import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

data = [[5, 7], [1, 3], [2, 4]]
assert merge(data) == [[1, 4], [5, 7]], "merge of overlapping intervals"
assert data == [[5, 7], [1, 3], [2, 4]], "argument after merge"
assert merge([[9, 6], [1, 2]]) == [[1, 2], [6, 9]], "merge with a reversed pair"
assert merge([[3, 5], [1, 3]]) == [[1, 5]], "merge of [3,5] and [1,3]"
assert total_covered([[1, 3], [3, 6], [10, 11]]) == 6, "covered length"
assert gaps([[2, 3], [5, 6]], 0, 10) == [(0, 2), (3, 5), (6, 10)], "gaps in [0, 10]"
assert intersect([[0, 10]], [[1, 2], [4, 5]]) == [[1, 2], [4, 5]], "intersect one long interval"
assert intersect([[1, 3], [5, 9]], [[2, 6]]) == [[2, 3], [5, 6]], "intersect two lists"
assert subtract([[0, 10]], [[2, 3], [5, 6]]) == [[0, 2], [3, 5], [6, 10]], "subtract"
assert max_overlap([[1, 4], [2, 5], [4, 6]]) == 3, "max overlap"
assert contains_point([[1, 3], [5, 7]], 4) is False, "point between intervals"
assert describe([[4, 6], [1, 2], [2, 3]]) == "[1, 3] U [4, 6]", "describe"
