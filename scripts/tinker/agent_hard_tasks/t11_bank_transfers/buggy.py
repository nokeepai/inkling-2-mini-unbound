"""Bank ledger: accounts with overdraft limits, transfers, history and statements.

Amounts are whole currency units (ints).
"""


class InsufficientFunds(Exception):
    pass


class Account:
    def __init__(self, owner, overdraft_limit=0, history=[]):
        self.owner = owner
        self.balance = 0
        self.overdraft_limit = overdraft_limit
        self.history = history

    def can_withdraw(self, amount):
        return self.balance - amount > -self.overdraft_limit

    def _log(self, kind, amount, memo):
        self.history.append((kind, amount, self.balance, memo))

    def deposit(self, amount, memo=""):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount
        self._log("deposit", amount, memo)

    def withdraw(self, amount, memo=""):
        if amount <= 0:
            raise ValueError("amount must be positive")
        if not self.can_withdraw(amount):
            raise InsufficientFunds(self.owner)
        self.balance -= amount
        self._log("withdraw", amount, memo)

    def available(self):
        return self.balance + self.overdraft_limit


class Bank:
    def __init__(self):
        self.accounts = {}
        self.transfers = []

    def open(self, owner, deposit=0, overdraft_limit=0):
        if owner in self.accounts:
            raise ValueError(f"account {owner} exists")
        acct = Account(owner, overdraft_limit=overdraft_limit)
        self.accounts[owner] = acct
        if deposit:
            acct.deposit(deposit, "opening deposit")
        return acct

    def transfer(self, src, dst, amount):
        """Move money between accounts; on failure neither account changes."""
        if src == dst:
            raise ValueError("cannot transfer to the same account")
        source, target = self.accounts[src], self.accounts[dst]
        target.deposit(amount, f"from {src}")
        source.withdraw(amount, f"to {dst}")
        self.transfers.append((src, dst, amount))

    def statement(self, owner):
        lines = []
        for kind, amount, balance, memo in self.accounts[owner].history:
            line = f"{kind} {amount} -> {balance}"
            if memo:
                line += f" ({memo})"
            lines.append(line)
        return lines

    def audit(self):
        """Owners whose history does not replay to their current balance."""
        bad = []
        for owner in sorted(self.accounts):
            acct = self.accounts[owner]
            running = 0
            for kind, amount, _, _ in acct.history:
                running += amount if kind == "deposit" else -amount
            if running != acct.balance:
                bad.append(owner)
        return bad

    def charge_fees(self, fee, minimum):
        """Charge `fee` to every account below `minimum` that can afford it; return who paid."""
        charged = []
        for owner in sorted(self.accounts):
            acct = self.accounts[owner]
            if acct.balance < minimum and acct.can_withdraw(fee):
                acct.withdraw(fee, "monthly fee")
                charged.append(owner)
        return charged

    def total_deposits(self):
        return sum(acct.balance for acct in self.accounts.values())

    def largest(self, n):
        ranked = sorted(self.accounts.values(), key=lambda a: (-a.balance, a.owner))
        return [(a.owner, a.balance) for a in ranked[:n]]

    def transfers_for(self, owner):
        return [t for t in self.transfers if owner in (t[0], t[1])]

    def net_flow(self, owner):
        """Money received by transfer minus money sent by transfer."""
        net = 0
        for src, dst, amount in self.transfers:
            if dst == owner:
                net += amount
            if src == owner:
                net -= amount
        return net
