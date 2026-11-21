"""Build a full merged checkpoint dir for SGLang's /update_weights_from_disk (partial loads are rejected).

  python sgl_merge.py <cand_dir|orig> <out_dir> [--base /srv/w/orig]
Copies the base model-*.safetensors, then overwrites only the candidate tensors' byte ranges in place
(same name, dtype BF16, shape: checked), and copies the small config/tokenizer files. `orig` -> out_dir is
a byte-identical copy (used for restore). Verifies sha256 of tensors.safetensors against sha256.txt.
"""
import hashlib
import json
import os
import shutil
import struct
import sys

from safetensors import safe_open

FORBIDDEN = ("experts", "w13_weight", "w2_weight", "_scale", "amax", "outer_scale")


def header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cand, out = sys.argv[1], sys.argv[2]
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "/srv/w/orig"
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(base):
        p = os.path.join(base, f)
        if os.path.isfile(p) and not f.startswith("model-"):
            shutil.copy2(p, os.path.join(out, f))
    shards = [f for f in os.listdir(base) if f.startswith("model-") and f.endswith(".safetensors")]
    for s in shards:
        shutil.copyfile(os.path.join(base, s), os.path.join(out, s))
    if cand == "orig":
        print("restore dir ready:", out)
        return
    tpath = os.path.join(cand, "tensors.safetensors")
    want = open(os.path.join(cand, "sha256.txt")).read().split()[0]
    assert sha256(tpath) == want, "candidate sha256 mismatch"
    hdrs = {s: header(os.path.join(out, s)) for s in shards}
    patched = 0
    with safe_open(tpath, "pt") as cf:
        for name in cf.keys():
            assert not any(t in name for t in FORBIDDEN), f"forbidden tensor {name}"
            t = cf.get_tensor(name)
            shard = next(s for s, (h, _) in hdrs.items() if name in h)
            h, off = hdrs[shard]
            meta = h[name]
            assert meta["dtype"] == "BF16" and list(t.shape) == meta["shape"], name
            a, b = meta["data_offsets"]
            data = t.contiguous().view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes()
            assert len(data) == b - a, name
            with open(os.path.join(out, shard), "r+b") as f:
                f.seek(off + a)
                f.write(data)
            patched += 1
    print(f"merged {patched} tensors into {out}")


if __name__ == "__main__":
    main()
