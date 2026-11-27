import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *
from fractions import Fraction

assert matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]], "2x2 product"
assert matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]) == [[58, 64], [139, 154]], "2x3 times 3x2"
assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]], "transpose"
assert identity(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "identity(3)"
assert determinant([[2, 1], [1, 1]]) == 1, "det [[2,1],[1,1]]"
assert determinant([[0, 2, 1], [1, 0, 0], [3, 1, 2]]) == -3, "det [[0,2,1],[1,0,0],[3,1,2]]"
assert inverse([[4, 7], [2, 6]]) == [[Fraction(3, 5), Fraction(-7, 10)], [Fraction(-1, 5), Fraction(2, 5)]], "inverse"
M = [[2, 1], [1, 1]]
assert matmul(M, inverse(M)) == identity(2), "M times its inverse"
assert power([[1, 1], [1, 0]], 5) == [[8, 5], [5, 3]], "fifth power of [[1,1],[1,0]]"
assert power([[3]], 0) == [[1]], "zeroth power"
assert solve([[0, 1], [2, 0]], [3, 4]) == [2, 3], "solve"
assert trace(add([[1, 2], [3, 4]], scale(identity(2), 2))) == 9, "trace"
