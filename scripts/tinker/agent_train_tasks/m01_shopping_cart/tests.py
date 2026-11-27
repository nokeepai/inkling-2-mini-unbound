import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import *
c = Cart()
assert c.count() == 0, "empty cart count"
assert c.subtotal() == 0, "empty subtotal"
assert cheapest_item(c) is None, "no cheapest in empty cart"
c.add("apple", 3, 200)
assert c.line_total("apple") == 500, "third apple half price"
assert bulk_price(10, 100) == 808, "ten units get bulk discount"
assert bulk_price(2, 100) == 200, "two units no discount"
assert bulk_price(0, 100) == 0, "zero units"
c.add("pear", 1, 150)
assert c.count() == 4, "count after pear"
assert c.subtotal() == 650, "subtotal"
assert cheapest_item(c) == "pear", "cheapest item"
assert savings(c) == 100, "bulk savings"
assert apply_coupon(1000, "SAVE10") == 900, "SAVE10 at threshold"
assert apply_coupon(999, "SAVE10") == 999, "SAVE10 below threshold"
assert apply_coupon(300, "FLAT5") == 0, "FLAT5 floors at zero"
assert apply_coupon(1000, "PCT50") == 500, "PCT50 allowed"
assert apply_coupon(1000, "PCT51") == 1000, "PCT51 rejected"
assert tax_on(650) == 52, "tax rounding"
assert c.total() == 702, "total with tax"
assert c.remove("apple", 1) == 1, "partial remove"
assert c.count() == 3, "count after remove"
assert c.remove("pear", 5) == 1, "over-remove returns actual"
assert "pear" not in c.items, "pear gone"
