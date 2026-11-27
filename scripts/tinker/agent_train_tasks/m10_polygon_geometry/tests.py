import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import *
sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
tri = [(0, 0), (4, 0), (0, 3)]
assert area(sq) == 16.0, "square area"
assert area(tri) == 6.0, "triangle area"
assert area([(0, 0), (1, 1)]) == 0.0, "degenerate area"
assert is_clockwise(sq) == False, "ccw square"
assert is_clockwise(list(reversed(sq))) == True, "cw square"
assert perimeter(tri) == 12.0, "3-4-5 perimeter"
assert perimeter([]) == 0.0, "empty perimeter"
assert boundary_points(sq) == 16, "square boundary points"
assert boundary_points(tri) == 8, "triangle boundary points"
assert interior_points(sq) == 9, "square interior points"
assert interior_points(tri) == 3, "triangle interior points"
assert bounding_box(tri) == (0, 0, 4, 3), "bounding box"
assert bounding_box([]) is None, "empty bounding box"
assert is_convex(sq) == True, "square convex"
assert is_convex([(0, 0), (4, 0), (2, 1), (4, 4), (0, 4)]) == False, "dented polygon"
assert is_convex([(0, 0), (1, 1)]) == False, "two points not convex"
assert contains(sq, (2, 2)) == True, "inside square"
assert contains(sq, (5, 2)) == False, "outside square"
assert contains(tri, (3, 2)) == False, "outside hypotenuse"
assert contains(tri, (1, 1)) == True, "inside triangle"
