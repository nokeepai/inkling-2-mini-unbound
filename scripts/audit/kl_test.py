"""Auditor: KL(orig || cand) and top-1 on N harmless prompts of a split, using harness/kl_eval.py UNCHANGED.

  FID_DIR=/srv/audit/kl_<split> FID_BIG=/srv/audit/klbig_<split> \
    python3 kl_test.py <harmless.jsonl> <n> gen|cache|eval [cand_dir|orig ...] [--bs 8] [--tag x]

Only the inputs differ from the inner loop: the prompt file (sealed test harmless, or val for the dry run) and
N_PROMPTS (1,000 for final runs, BRIEF §4). Prompt choice = kl_eval.pick_ids (random.Random(1234).sample over the file).
Everything else is kl_eval's: orig greedy continuations at effort 0 (<=256 tokens, left-padded bs 100), the fp16
full-vocab log-prob cache (bs 8, sorted, right padding), candidates scored with the cache's exact batching,
act_quant OFF (W4A16), token-pooled KL and top-1 with prompt-bootstrap CIs (10k, seed 1234).
Results append to $FID_DIR/kl_results.jsonl.
"""
from __future__ import annotations

import os
import shutil
import sys

HARNESS = os.environ.get("AUDIT_HARNESS", "/srv/audit/harness")


def main() -> None:
    src, n, rest = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    fid = os.environ["FID_DIR"]
    os.makedirs(f"{fid}/data", exist_ok=True)
    os.makedirs(os.environ["FID_BIG"], exist_ok=True)
    dst = f"{fid}/data/harmless.jsonl"
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
    sys.path.insert(0, HARNESS)
    import kl_eval as K  # noqa: E402  (module reads FID_DIR / FID_BIG at import)
    assert K.DATA == dst, (K.DATA, dst)
    K.N_PROMPTS = n
    sys.argv = ["kl_eval.py", *rest]
    K.main()


if __name__ == "__main__":
    main()
