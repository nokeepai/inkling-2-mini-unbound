#!/usr/bin/env python3
"""Seal the harness: sha256 of every file under harness/ and data/ (sorted), plus the sealed ciphertext.
Writes harness/SEALED (the manifest + one combined hash) and prints the combined hash.
Excludes harness/SEALED itself and __pycache__ / runtime-log files."""
import hashlib, os, json
from datetime import datetime, timezone

R = "/Users/nikshepsvn/Code/inkling-abl"
SKIP_DIRS = {"__pycache__", ".sync"}
SKIP_FILES = {"SEALED"}


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


entries = []
for top in ("harness", "data"):
    for root, dirs, files in os.walk(f"{R}/{top}"):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for fn in sorted(files):
            if fn in SKIP_FILES or fn.endswith((".log", ".pyc")):
                continue
            p = os.path.join(root, fn)
            entries.append((os.path.relpath(p, R), sha(p)))
entries.append(("sealed.tar.enc", sha(f"{R}/sealed.tar.enc")))
combined = hashlib.sha256("\n".join(f"{p} {h}" for p, h in entries).encode()).hexdigest()
with open(f"{R}/harness/SEALED", "w") as f:
    f.write(f"# Harness seal, {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    f.write("# Owner approvals: state/approvals.md (judge gate option 4, hygiene redefined, probe regex widened, crisis report-only)\n")
    f.write(f"COMBINED_SHA256 {combined}\nFILES {len(entries)}\n")
    for p, h in entries:
        f.write(f"{h}  {p}\n")
print(combined, len(entries))
