"""Append screen queue rows at a given priority for existing c3 candidates.  bump.py <priority> <cid>..."""
import sys

sys.argv = [sys.argv[0], "1", "1"] + sys.argv[1:]
sys.path.insert(0, __import__("os").path.dirname(__file__))
import search as S  # noqa: E402

prio = float(sys.argv[3])
for cid in sys.argv[4:]:
    S.enqueue(cid, "screen", prio)
    print("bumped", cid, prio)
