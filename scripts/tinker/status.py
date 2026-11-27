"""One-line W1 progress: last step + key metrics per arm, eval completion, judge spend.
Appends to ~/.inkling-abl/tinker_lab/status.txt and prints the line.

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/status.py w1a w1b w1c w1d
"""
from __future__ import annotations

import json
import sys
import time

import lab_common as lc


def arm_status(tag: str) -> str:
    log = lc.LAB_DIR / "logs" / f"train_{tag}.log"
    if not log.exists():
        return f"{tag}:nolog"
    last, errors = None, 0
    for line in log.read_text().splitlines():
        r = json.loads(line)
        if r["event"] == "step":
            last = r
        errors += r["event"] in ("error", "probe_error")
    stdout = (lc.LAB_DIR / "logs" / f"train_{tag}.stdout")
    tb = "Traceback" in stdout.read_text() if stdout.exists() else False
    evals = sum((lc.LAB_DIR / "eval" / f"{tag}_step{s}.summary.json").exists() for s in (20, 40))
    if last is None:
        return f"{tag}:starting{' TRACEBACK' if tb else ''}"
    return (f"{tag}:s{last['step'] + 1} R={last['reward']:.2f} cap={last['cap_hit']:.2f} "
            f"think={last['think_tok_mean']:.0f} {last['sec_step']:.0f}s/step evals={evals}/2"
            f"{' TRACEBACK' if tb else ''}")


def main() -> None:
    import judge

    line = (time.strftime("%H:%M ") + " | ".join(arm_status(t) for t in sys.argv[1:])
            + f" | judge ${judge.spent():.3f}")
    with (lc.LAB_DIR / "status.txt").open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()
