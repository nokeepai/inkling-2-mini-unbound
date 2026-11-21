"""Enqueue the full stage for c3 candidates outside the Optuna loop.  promote_full.py <cid>..."""
import os
import sys

args = sys.argv[1:]
sys.argv = [sys.argv[0], "1", "1"]
sys.path.insert(0, os.path.dirname(__file__))
import search as S  # noqa: E402

for cid in args:
    S.enqueue(cid, "full", 7.0)
    print("full enqueued", cid)
