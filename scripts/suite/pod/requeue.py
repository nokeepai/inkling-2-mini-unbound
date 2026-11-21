"""Pod: re-enqueue (try+1) every queue job whose current key is not done. Run only while no worker is alive."""
import fcntl
import json
import os
import sys

sys.path.insert(0, "/srv/suite")
import suite_worker as W  # noqa: E402

done = {json.loads(l)["key"] for l in open(W.C) if '"done"' in l} if os.path.exists(W.C) else set()
n = 0
with open(W.Q, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    for r in W.queue():
        if W.key(r) not in done:
            r = dict(r, **{"try": r.get("try", 0) + 1})
            f.write(json.dumps(r) + "\n")
            n += 1
print("requeued", n)
