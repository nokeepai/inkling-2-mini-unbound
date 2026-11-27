import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

inv = Inventory()
inv.receive("bolt", 10, 2.0)
assert inv.average_cost("bolt") == 2.0, "average cost after first receipt"
inv.receive("bolt", 30, 4.0)
assert inv.average_cost("bolt") == 3.5, "average cost after second receipt"
assert inv.total_value() == 140.0, "stock value"
assert inv.ship("bolt", 15) == 52.5, "cost of a partial shipment"
assert inv.quantity("bolt") == 25, "remaining units"
assert inv.ship("bolt", 25) == 87.5, "cost of shipping the rest"
assert inv.quantity("bolt") == 0, "nothing left"
inv.receive("bolt", 4, 5.0)
assert inv.average_cost("bolt") == 5.0, "average cost after restocking an empty sku"
try:
    inv.ship("bolt", 5)
    raised = False
except InsufficientStock:
    raised = True
assert raised, "shipping more than on hand"
inv.receive("nut", 3, 0.5)
inv.receive("washer", 8, 0.1)
inv.tag("nut", "metric")
assert inv.with_tag("metric") == ["nut"], "skus tagged metric"
assert inv.items["washer"]["tags"] == [], "tags of washer"
inv.receive("gear", 2, 7.0, tags=["spare"])
assert inv.with_tag("spare") == ["gear"], "skus tagged spare"
assert inv.low_stock(4) == ["gear", "nut", "bolt"], "low stock order"
assert inv.cost_of_goods_shipped() == 140.0, "total cost of goods shipped"
assert inv.restock_plan({"nut": 10, "washer": 5, "pin": 2}) == {"nut": 7, "pin": 2}, "restock plan"
