import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

assert compare("1.2.10", "1.2.9") == 1, "1.2.10 vs 1.2.9"
assert compare("1.0.0-alpha", "1.0.0") == -1, "1.0.0-alpha vs 1.0.0"
assert compare("1.0.0+build.5", "1.0.0") == 0, "1.0.0+build.5 vs 1.0.0"
assert compare("1.0.0-alpha.1", "1.0.0-alpha.beta") == -1, "1.0.0-alpha.1 vs 1.0.0-alpha.beta"
assert compare("v2.0", "2.0.0") == 0, "v2.0 vs 2.0.0"
order = ["1.0.0", "1.0.0-rc.1", "1.0.0-beta.11", "1.0.0-beta.2", "1.0.0-beta", "1.0.0-alpha.1", "1.0.0-alpha", "0.9"]
assert sort_versions(order) == ["0.9", "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"], "sorted order"
assert satisfies("1.9.3", "^1.2") is True, "1.9.3 satisfies ^1.2"
assert satisfies("2.0", "^1.2") is False, "2.0 satisfies ^1.2"
assert satisfies("1.2.9", "~1.2.3") is True, "1.2.9 satisfies ~1.2.3"
assert satisfies("1.3", "~1.2.3") is False, "1.3 satisfies ~1.2.3"
assert satisfies("1.4", ">=1.4.0") and not satisfies("1.4.0-rc.1", ">=1.4"), "comparison operators"
assert max_satisfying(["1.4.0", "2.0.0", "1.10.1", "1.10.1-rc.1", "0.9"], "^1.0") == "1.10.1", "max satisfying ^1.0"
assert max_satisfying(["0.1", "0.2"], ">=1.0") is None, "max satisfying >=1.0"
assert bump("1.9", "minor") == "1.10.0" and bump("1.2.3-rc.1", "patch") == "1.2.4", "bump"
