"""Semantic-version parsing, comparison, sorting and simple range matching."""
import functools


class Version:
    def __init__(self, release, prerelease=(), build=""):
        self.release = tuple(release)
        self.prerelease = tuple(prerelease)
        self.build = build

    def __repr__(self):
        return f"Version({str(self)!r})"

    def __str__(self):
        text = ".".join(str(p) for p in self.release)
        if self.prerelease:
            text += "-" + ".".join(str(p) for p in self.prerelease)
        if self.build:
            text += "+" + self.build
        return text


def parse(text):
    """Parse '1.2.3-beta.2+build.7' style strings (a leading 'v' is allowed)."""
    text = text.strip()
    if text.startswith("v"):
        text = text[1:]
    text, _, build = text.partition("+")
    core, _, pre = text.partition("-")
    release = [int(part) for part in core.split(".")]
    prerelease = []
    if pre:
        for ident in pre.split("."):
            prerelease.append(ident)
    return Version(release, prerelease, build)


def _as_version(v):
    return parse(v) if isinstance(v, str) else v


def _pad(release, length):
    return tuple(release) + (0,) * (length - len(release))


def _compare_identifiers(a, b):
    a_num, b_num = isinstance(a, int), isinstance(b, int)
    if a_num != b_num:
        return -1 if a_num else 1
    return (a > b) - (a < b)


def compare(a, b):
    """-1, 0 or 1 by semantic-version precedence; build metadata is ignored."""
    a, b = _as_version(a), _as_version(b)
    length = max(len(a.release), len(b.release))
    ra, rb = _pad(a.release, length), _pad(b.release, length)
    if ra != rb:
        return -1 if ra < rb else 1
    if not a.prerelease or not b.prerelease:
        return (not a.prerelease) - (not b.prerelease)
    for x, y in zip(a.prerelease, b.prerelease):
        result = _compare_identifiers(x, y)
        if result:
            return result
    return (len(a.prerelease) > len(b.prerelease)) - (len(a.prerelease) < len(b.prerelease))


def sort_versions(versions, reverse=False):
    return sorted(versions, key=functools.cmp_to_key(compare), reverse=reverse)


def _bounds(spec):
    """(lower, lower_inclusive, upper) for caret and tilde specs."""
    base = parse(spec[1:])
    release = _pad(base.release, 3)
    if spec[0] == "^":
        upper = Version([release[0] + 1])
    else:
        upper = Version([release[0], release[1] + 1])
    return base, upper


def satisfies(version, spec):
    """Supports ^X.Y.Z, ~X.Y.Z, >=, <=, >, <, = and bare versions (exact match)."""
    v = _as_version(version)
    spec = spec.strip()
    if spec[0] in "^~":
        lower, upper = _bounds(spec)
        return compare(v, lower) >= 0 and compare(v, upper) <= 0
    for op in (">=", "<=", ">", "<", "="):
        if spec.startswith(op):
            result = compare(v, spec[len(op):].strip())
            return {">=": result >= 0, "<=": result <= 0, ">": result > 0,
                    "<": result < 0, "=": result == 0}[op]
    return compare(v, spec) == 0


def max_satisfying(versions, spec):
    matches = [v for v in versions if satisfies(v, spec)]
    return sort_versions(matches)[-1] if matches else None


def bump(version, part):
    """Next version after bumping 'major', 'minor' or 'patch' (pre-release dropped)."""
    major, minor, patch = _pad(_as_version(version).release, 3)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown part {part}")
