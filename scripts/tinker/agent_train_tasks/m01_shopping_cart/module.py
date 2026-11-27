"""Shopping cart with bulk pricing, coupon codes and sales tax (all in cents)."""

TAX_RATE_PERCENT = 8


class Cart:
    def __init__(self):
        self.items = {}
        self.prices = {}

    def add(self, name, qty, price_cents=100):
        if qty <= 0:
            raise ValueError("quantity must be positive")
        self.items[name] = self.items.get(name, 0) + qty
        self.prices[name] = price_cents

    def remove(self, name, qty):
        """Remove up to qty units; returns how many were actually removed."""
        current = self.items.get(name, 0)
        if qty >= current:
            self.items.pop(name, None)
            self.prices.pop(name, None)
            return current
        self.items[name] = current - qty
        return qty

    def count(self):
        total = 0
        for name in self.items:
            total += self.items[name]
        return total

    def line_total(self, name):
        qty = self.items.get(name, 0)
        price = self.prices.get(name, 0)
        return bulk_price(qty, price)

    def subtotal(self):
        return sum(self.line_total(n) for n in self.items)

    def total(self, coupon=""):
        sub = self.subtotal()
        discounted = apply_coupon(sub, coupon)
        return discounted + tax_on(discounted)


def bulk_price(qty, unit_cents):
    """Every 3rd unit is half price; 10 or more units get 5% off the whole line."""
    half = qty // 3
    full = qty - half
    line = full * unit_cents + half * (unit_cents // 2)
    if qty >= 10:
        line = line - line * 5 // 100
    return line


def apply_coupon(amount, code):
    if code == "SAVE10" and amount >= 1000:
        return amount - amount // 10
    if code == "FLAT5":
        return max(0, amount - 500)
    if code.startswith("PCT") and len(code) == 5 and code[3:].isdigit():
        pct = int(code[3:])
        if 0 < pct and pct <= 50:
            return amount - amount * pct // 100
    return amount


def tax_on(amount):
    # round half up to the nearest cent
    return (amount * TAX_RATE_PERCENT + 50) // 100


def cheapest_item(cart):
    best = None
    for name in sorted(cart.items):
        price = cart.prices[name]
        if best is None or price < cart.prices[best]:
            best = name
    return best


def savings(cart):
    """Cents saved by bulk pricing relative to paying full price for every unit."""
    full = 0
    for name in cart.items:
        full += cart.items[name] * cart.prices[name]
    return full - cart.subtotal()
