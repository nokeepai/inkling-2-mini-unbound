#!/usr/bin/env python3
"""Record gate 2/3 status in state/phase.json."""
import json

P = "/Users/nikshepsvn/Code/inkling-abl/state/phase.json"
d = json.load(open(P))
d["gate1"] = {"status": "PENDING_SEAL", "notes": "harness re-seal after loop detector validation (owner condition)"}
d["gate2"] = {"status": "PASSED", "time": "2026-11-21T12:20Z",
              "notes": "directions, leak, attribution (MoE share 0.55-0.93), controllers shared"}
d["gate3"] = {"status": "PROVISIONAL", "time": "2026-11-21T15:28Z", "finalists": ["c1h_svd1_s2", "c1t013"],
              "notes": "slot B open to a lower-KL c1c point until 15:50Z (owner: best model)",
              "report": "reports/gate3_finalists.md"}
json.dump(d, open(P, "w"), indent=1)
print("ok")
