"""List the account's checkpoints with sizes, and delete named ones to free storage.

Why: the account's checkpoint storage has a free limit; once it is hit, every save closes its
training client. Trainer-state checkpoints ("weights/") of finished arms are never resumed, so
they are the first thing to delete; sampler checkpoints are what evals use.

  $PY ckpt_admin.py list
  $PY ckpt_admin.py delete --match w1b --match w1c            # dry run: prints what would go
  $PY ckpt_admin.py delete --match w1b --kind training --yes  # really delete
"""
from __future__ import annotations

import argparse
import re

import lab_common as lc


def all_checkpoints(rc):
    out = []
    offset = 0
    while True:
        runs = rc.list_training_runs(limit=100, offset=offset).result()
        for run in runs.training_runs:
            for c in rc.list_checkpoints(run.training_run_id).result().checkpoints:
                out.append((run, c))
        offset += len(runs.training_runs)
        if not runs.training_runs or offset >= runs.cursor.total_count:
            return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "delete"])
    ap.add_argument("--match", action="append", default=[], help="regex on the checkpoint id (repeatable)")
    ap.add_argument("--kind", choices=["training", "sampler", "any"], default="any")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    lc.load_key()
    import tinker

    rc = tinker.ServiceClient().create_rest_client()
    rows = all_checkpoints(rc)
    total = 0
    for run, c in sorted(rows, key=lambda rc_: str(rc_[1].time)):
        size = getattr(c, "size_bytes", None) or 0
        total += size
        hit = any(re.search(m, c.checkpoint_id) for m in args.match) and \
            (args.kind == "any" or c.checkpoint_type == args.kind)
        if args.cmd == "list" or hit:
            print(f"{'DEL ' if hit and args.cmd == 'delete' else ''}{c.checkpoint_type:9s} {size / 1e6:8.1f} MB  "
                  f"{str(c.time)[:16]}  {run.base_model.split('/')[-1][:28]:28s} {c.tinker_path}")
        if hit and args.cmd == "delete" and args.yes:
            rc.delete_checkpoint_from_tinker_path(c.tinker_path).result()
    print(f"{len(rows)} checkpoints, {total / 1e9:.2f} GB total")


if __name__ == "__main__":
    main()
