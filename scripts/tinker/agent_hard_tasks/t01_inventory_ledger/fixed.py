"""Inventory ledger with weighted-average costing, tags and a stock report."""


class InsufficientStock(Exception):
    """Raised when a shipment asks for more units than are on hand."""


def _weighted_average(old_qty, old_cost, new_qty, new_cost):
    total = old_qty + new_qty
    if total == 0:
        return 0.0
    return (old_qty * old_cost + new_qty * new_cost) / total


def _money(amount):
    return round(amount + 0.0, 2)


class Inventory:
    def __init__(self):
        self.items = {}
        self.history = []

    def _record(self, kind, sku, qty, amount):
        self.history.append({"kind": kind, "sku": sku, "qty": qty, "amount": _money(amount)})

    def receive(self, sku, qty, unit_cost, tags=None):
        """Add stock at a unit cost; the SKU's average cost is re-weighted."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        if tags is None:
            tags = []
        item = self.items.get(sku)
        if item is None:
            item = {"qty": 0, "avg_cost": 0.0, "tags": tags}
            self.items[sku] = item
        else:
            for label in tags:
                if label not in item["tags"]:
                    item["tags"].append(label)
        item["avg_cost"] = _weighted_average(item["qty"], item["avg_cost"], qty, unit_cost)
        item["qty"] += qty
        self._record("receive", sku, qty, qty * unit_cost)
        return item["qty"]

    def tag(self, sku, label):
        item = self.items[sku]
        if label not in item["tags"]:
            item["tags"].append(label)

    def untag(self, sku, label):
        item = self.items[sku]
        if label in item["tags"]:
            item["tags"].remove(label)

    def ship(self, sku, qty):
        """Remove stock and return the cost of the goods shipped."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        item = self.items.get(sku)
        if item is None or item["qty"] < qty:
            raise InsufficientStock(sku)
        item["qty"] -= qty
        cogs = qty * item["avg_cost"]
        if item["qty"] == 0:
            item["avg_cost"] = 0.0
        self._record("ship", sku, qty, cogs)
        return _money(cogs)

    def quantity(self, sku):
        item = self.items.get(sku)
        return 0 if item is None else item["qty"]

    def average_cost(self, sku):
        return _money(self.items[sku]["avg_cost"])

    def total_value(self):
        return _money(sum(it["qty"] * it["avg_cost"] for it in self.items.values()))

    def low_stock(self, threshold):
        """SKUs at or below threshold, lowest quantity first, ties by SKU."""
        hits = [(it["qty"], sku) for sku, it in self.items.items() if it["qty"] <= threshold]
        return [sku for _, sku in sorted(hits)]

    def with_tag(self, label):
        return sorted(sku for sku, it in self.items.items() if label in it["tags"])

    def history_for(self, sku):
        return [h for h in self.history if h["sku"] == sku]

    def cost_of_goods_shipped(self):
        return _money(sum(h["amount"] for h in self.history if h["kind"] == "ship"))

    def restock_plan(self, targets):
        """Units to order so each SKU reaches its target level (only positive amounts)."""
        plan = {}
        for sku, target in sorted(targets.items()):
            missing = target - self.quantity(sku)
            if missing > 0:
                plan[sku] = missing
        return plan

    def report(self):
        lines = []
        for sku in sorted(self.items):
            it = self.items[sku]
            value = it["qty"] * it["avg_cost"]
            labels = ",".join(sorted(it["tags"])) or "-"
            lines.append(f"{sku:<8}{it['qty']:>5}{it['avg_cost']:>9.2f}{value:>10.2f}  {labels}")
        lines.append(f"{'TOTAL':<8}{'':>5}{'':>9}{self.total_value():>10.2f}")
        return "\n".join(lines)


def build_inventory(rows):
    """Create an inventory from (sku, qty, unit_cost) rows."""
    inv = Inventory()
    for sku, qty, unit_cost in rows:
        inv.receive(sku, qty, unit_cost)
    return inv
