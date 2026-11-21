"""Pod-side check that a candidate's extra tensors (e.g. router mlp.gate.weight) take effect in SGLang:
swap A, read the probe's top-20 logprobs, restore; swap B, read, restore. B != A means B's extra tensors changed the
forward pass. Uses sgl_eval's swap/restore (verified), per-GPU dirs like fleet_eval.
  python3 /srv/harness/probe_pair.py --gpu 7 <candA> <candB>     (both already in /srv/cands/<id>__g7 with tensors)"""
import json
import sys

sys.path.insert(0, "/srv/harness")
import fleet_eval as F  # noqa: E402

gpu = int(sys.argv[sys.argv.index("--gpu") + 1])
a, b = [x for x in sys.argv[1:] if not x.startswith("--") and x != str(gpu)][:2]
F.CONTAINER, F.SLOT = f"sgl{gpu}", f"__g{gpu}"
F.S.PORT = F.gpu_port(gpu)
base = F.ensure_orig()
out = {}
for c in (a, b):
    F.prepare(c)
    sw = F.S.swap(c + F.SLOT)
    out[c] = sw["swapped"]
    F.S.restore(base)
    import shutil
    shutil.rmtree(f"/srv/w/merged_{c}{F.SLOT}", ignore_errors=True)
print(json.dumps({"orig_top5": base[:5], a: out[a][:5], b: out[b][:5], "a_ne_orig": out[a] != base,
                  "b_ne_orig": out[b] != base, "b_ne_a": out[b] != out[a], "restored_ok": True}))
