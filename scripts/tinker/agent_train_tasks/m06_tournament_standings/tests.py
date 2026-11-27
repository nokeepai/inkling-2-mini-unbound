import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import *
assert parse_result("Lions 2-1 Tigers") == ("Lions", 2, 1, "Tigers"), "parse simple"
assert parse_result("Red Sox 0-0 Blue Jays") == ("Red Sox", 0, 0, "Blue Jays"), "parse multiword"
assert fixtures(["a", "b", "c"]) == [("a", "b"), ("a", "c"), ("b", "c")], "fixtures"
assert fixtures(["a"]) == [], "single team fixtures"
t = Table(["A", "B", "C"])
assert t.standings() == ["A", "B", "C"], "initial standings by name"
assert unbeaten(t) == [], "nobody has played"
t.record_line("A 2-0 B")
t.record_line("B 1-1 C")
t.record_line("C 3-2 A")
assert t.points("C") == 4, "C points"
assert t.points("A") == 3, "A points"
assert t.points("B") == 1, "B points"
assert t.played("B") == 2, "B played"
assert t.goal_diff("A") == 1, "A goal diff"
assert t.goal_diff("B") == -2, "B goal diff"
assert t.standings() == ["C", "A", "B"], "standings"
assert t.position("B") == 3, "position last"
assert t.leader_gap("A") == 1, "gap to leader"
assert clinched(t, "C", 0) == True, "clinched with no games left"
assert clinched(t, "C", 1) == False, "not clinched with games left"
assert unbeaten(t) == ["C"], "unbeaten"
t.record("A", 1, 1, "C")
assert t.standings() == ["C", "A", "B"], "standings after draw"
