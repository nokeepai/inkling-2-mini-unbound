"""Mac: Jev-judge every row of a gens jsonl (prompt + answer); resumable. Uses harness/judge_jev.judge_one unchanged.

  python3 judge_file.py <gens.jsonl> <out.jsonl> [conc=16]

Out rows: {"id", "k", "sys", "effort", **judge_one(...)}; rows with errors are retried on the next run.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.expanduser("~/Code/inkling-abl/harness"))
from judge_jev import judge_one  # noqa: E402


def key(r: dict) -> tuple:
    return (r["id"], r.get("k", 0), r.get("sys", 0), r.get("effort", 0))


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    conc = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            if r.get("error") is None:
                done.add(key(r))
    rows = [json.loads(l) for l in open(src)]
    todo = [r for r in rows if key(r) not in done and not r.get("cap_hit")]
    print(f"{len(todo)} to judge ({len(rows)} rows, {len(done)} done)", flush=True)
    lock = threading.Lock()
    f = open(out, "a")
    n = [0]

    def one(r: dict) -> None:
        j = judge_one(r["prompt"], r["answer"])
        with lock:
            f.write(json.dumps({"id": r["id"], "k": r.get("k", 0), "sys": r.get("sys", 0),
                                "effort": r.get("effort", 0), **j}) + "\n")
            f.flush()
            n[0] += 1
            if n[0] % 500 == 0:
                print("judged", n[0], flush=True)

    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, todo))
    print("JUDGE_DONE", out, n[0], flush=True)


if __name__ == "__main__":
    main()
