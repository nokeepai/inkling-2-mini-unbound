import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

t = format_table(["name", "qty"], [["bolt", 3], ["nut", 12]], align=["left", "right"])
assert t.splitlines() == ["| name | qty |", "| :--- | --: |", "| bolt |   3 |", "| nut  |  12 |"], "\n" + t
rows = [["a|b", "x"], ["c"]]
t = format_table(["key", "val"], rows)
assert t.splitlines() == ["| key  | val |", "| ---- | --- |", "| a\\|b | x   |", "| c    |     |"], "\n" + t
assert rows == [["a|b", "x"], ["c"]], "rows after format_table"
t = format_table(["id", "t"], [[1, "ab"], [22, None]], align=["center", "center"])
assert t.splitlines() == ["| id  |  t  |", "| :-: | :-: |", "|  1  | ab  |", "| 22  |     |"], "\n" + t
md = format_table(["k", "v"], [["p|q", 1], ["r"]], align=["right", None])
assert parse_table(md) == (["k", "v"], [["p|q", "1"], ["r", ""]], ["right", None]), "parse back"
t = from_dicts([{"sku": "A1", "n": 2}, {"sku": "B22", "n": None}])
assert t.splitlines()[0] == "| n   | sku |" and t.splitlines()[3] == "|     | B22 |", "\n" + t
data = [["x", "2"], ["y", "3.5"]]
assert column_totals(["item", "cost"], data, {"cost"})[-1] == ["total", 5.5], "totals row"
