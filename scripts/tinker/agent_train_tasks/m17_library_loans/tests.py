import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import *
lib = Library()
lib.add_title("Dune", 2)
lib.add_title("Emma")
assert lib.available("Dune") == 2, "copies available"
assert lib.available("Zorro") == 0, "unknown title"
assert lib.checkout("ann", "Dune", 0) == True, "checkout"
assert lib.checkout("ann", "Dune", 0) == False, "same title twice"
assert lib.checkout("bob", "Dune", 1) == True, "second copy"
assert lib.checkout("cy", "Dune", 1) == False, "no copies left"
assert lib.due_soon(11) == [("ann", "Dune")], "due soon window"
assert lib.renew("ann", "Dune", 14) == True, "renew on due date"
assert lib.renew("ann", "Dune", 20) == True, "second renewal"
assert lib.renew("ann", "Dune", 21) == False, "renewal limit"
assert lib.overdue(15) == [], "due day not overdue"
assert lib.overdue(16) == [("bob", "Dune")], "overdue"
assert lib.give_back("bob", "Dune", 19) == 100, "late fine"
assert lib.checkout("bob", "Emma", 19) == False, "blocked by fine"
assert lib.pay("bob", 150) == 50, "change returned"
assert lib.checkout("bob", "Emma", 19) == True, "allowed after paying"
assert late_fine(10, 10) == 0, "on time"
assert late_fine(10, 11) == 25, "one day late"
assert projected_fine(0, 50, 50) == 1000, "fine capped"
assert lib.give_back("ann", "Dune", 34) == 0, "returned on time"
assert lib.loans_of("ann") == [], "no loans left"
