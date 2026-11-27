"""Library loan tracker with day-number due dates, renewals and capped late fines."""

LOAN_DAYS = 14
MAX_RENEWALS = 2
FINE_PER_DAY = 25
FINE_CAP = 1000
MAX_LOANS = 3


def late_fine(due, returned):
    days = returned - due
    if days <= 0:
        return 0
    return min(days * FINE_PER_DAY, FINE_CAP)


class Library:
    def __init__(self):
        self.copies = {}
        self.loans = {}  # (member, title) -> [due_day, renewals]
        self.fines = {}

    def add_title(self, title, copies=1):
        self.copies[title] = self.copies.get(title, 0) + copies

    def available(self, title):
        out = sum(1 for (_, t) in self.loans if t == title)
        return self.copies.get(title, 0) - out

    def loans_of(self, member):
        return sorted(t for (m, t) in self.loans if m == member)

    def checkout(self, member, title, today):
        if self.available(title) <= 0:
            return False
        if len(self.loans_of(member)) >= MAX_LOANS:
            return False
        # members with unpaid fines or the same title already out are refused
        if (member, title) in self.loans or self.fines.get(member, 0) > 0:
            return False
        self.loans[(member, title)] = [today + LOAN_DAYS, 0]
        return True

    def renew(self, member, title, today):
        loan = self.loans.get((member, title))
        if loan is None or loan[1] >= MAX_RENEWALS or today > loan[0]:
            return False
        loan[0] = today + LOAN_DAYS
        loan[1] += 1
        return True

    def give_back(self, member, title, today):
        due, _ = self.loans.pop((member, title))
        fine = late_fine(due, today)
        if fine > 0:
            self.fines[member] = self.fines.get(member, 0) + fine
        return fine

    def pay(self, member, amount):
        """Pay toward fines; returns change owed back to the member."""
        owed = self.fines.get(member, 0)
        paid = min(owed, amount)
        self.fines[member] = owed - paid
        return amount - paid

    def overdue(self, today):
        return sorted(k for k, v in self.loans.items() if today > v[0])

    def due_soon(self, today, within=3):
        """Loans due between today and today + within, inclusive."""
        out = []
        for key, (due, _) in sorted(self.loans.items()):
            if today <= due and due <= today + within:
                out.append(key)
        return out


def projected_fine(due, today, extra_days):
    return late_fine(due, today + extra_days)
