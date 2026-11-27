import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

bank = Bank()
alice = bank.open("alice", 100)
bob = bank.open("bob", 20, overdraft_limit=50)
carol = bank.open("carol")
bank.transfer("bob", "alice", 70)
assert (alice.balance, bob.balance) == (170, -50), "balances after bob sends alice 70"
try:
    bank.transfer("carol", "alice", 5)
    failed = False
except InsufficientFunds:
    failed = True
assert failed and alice.balance == 170 and carol.balance == 0, "balances after carol tries to send 5"
assert carol.history == [], "carol's history"
assert bank.statement("alice") == ["deposit 100 -> 100 (opening deposit)", "deposit 70 -> 170 (from bob)"], "alice's statement"
assert bank.audit() == [], "audit"
dave = bank.open("dave", 10)
assert bank.charge_fees(fee=10, minimum=50) == ["dave"], "fee run"
assert dave.balance == 0 and dave.available() == 0, "dave after fees"
assert bank.largest(2) == [("alice", 170), ("carol", 0)], "largest accounts"
assert bank.net_flow("bob") == -70 and bank.transfers_for("carol") == [], "transfer log"
