"""Patch the base BF16 GGUF with our 20 edited wo_ud tensors.  Runs ON the pod.

  python3 patch_gguf.py <base.gguf> <orig_safetensors_dir> <built tensors.safetensors> <name_map.json> <out.gguf>

name_map.json: {checkpoint_name: gguf_tensor_name}.  For each pair we first PROVE the mapping: the original checkpoint
tensor's BF16 bytes must equal the base GGUF tensor's bytes exactly (same dtype, same element order).  Only then is the
edited tensor's bytes written in its place.

Output layout: header bytes copied from the base with only the general.name value spliced, re-padded to the alignment,
then the base data section verbatim with the 20 regions overwritten.  Tensor-info offsets are relative to the data
section, so they stay valid unchanged.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import sys

import numpy as np
import torch
from gguf import GGUFReader, GGUFValueType
from safetensors import safe_open
from safetensors.torch import load_file

NEW_NAME = "Inkling-2-Mini-OpenUncensored-wo_ud-patched"


def bf16_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def orig_tensor(orig_dir: str, name: str) -> torch.Tensor:
    wm = json.load(open(os.path.join(orig_dir, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(orig_dir, wm[name]), "pt") as f:
        return f.get_tensor(name)


def new_header(base: str, rd: GGUFReader) -> bytes:
    """Base header (everything before the data section, minus trailing padding) with general.name replaced."""
    field = rd.fields.get("general.name")
    if field is None:
        raise SystemExit("base has no general.name; refusing to guess a header edit")
    assert field.types == [GGUFValueType.STRING]
    with open(base, "rb") as f:
        raw = f.read(rd.data_offset)
    # field.parts: [key_len, key, type, str_len, str]; locate the value's length word by its absolute offset
    str_len_part, str_part = field.parts[3], field.parts[4]
    val_start = field.offset + sum(p.nbytes for p in field.parts[:3])
    assert int(str_len_part[0]) == len(bytes(str_part)), "unexpected field layout"
    old_val_end = val_start + 8 + len(bytes(str_part))
    print(f"general.name: {bytes(str_part).decode()!r} -> {NEW_NAME!r}")
    enc = NEW_NAME.encode()
    # end of tensor infos = last tensor-info field end; data padding follows
    ti_end = max(t.field.offset + sum(p.nbytes for p in t.field.parts) for t in rd.tensors)
    hdr = raw[:val_start] + struct.pack("<Q", len(enc)) + enc + raw[old_val_end:ti_end]
    align = int(rd.fields["general.alignment"].parts[-1][0]) if "general.alignment" in rd.fields else 32
    pad = (-len(hdr)) % align
    return hdr + b"\x00" * pad


def main() -> None:
    base, orig_dir, built, map_path, out = sys.argv[1:]
    name_map: dict[str, str] = json.load(open(map_path))
    assert len(name_map) == 20, len(name_map)
    rd = GGUFReader(base)
    by_name = {t.name: t for t in rd.tensors}
    edited = load_file(built)
    assert set(edited) == set(name_map), "built tensors and name map disagree"

    patches: list[tuple[int, bytes]] = []  # (offset within data section, bytes)
    for ck, gn in sorted(name_map.items()):
        t = by_name[gn]
        gbytes = np.asarray(t.data).tobytes()
        ob = bf16_bytes(orig_tensor(orig_dir, ck))
        if gbytes != ob:
            raise SystemExit(f"MAPPING NOT 1:1: {ck} -> {gn} (base bytes differ from checkpoint; shape {t.shape} type {t.tensor_type})")
        eb = bf16_bytes(edited[ck])
        assert len(eb) == len(gbytes) == t.n_bytes, (ck, len(eb), len(gbytes))
        patches.append((t.data_offset - rd.data_offset, eb))
        print(f"proved 1:1 and staged {ck} -> {gn} {list(t.shape)} {t.tensor_type.name}")

    hdr = new_header(base, rd)
    tmp = out + ".part"
    with open(base, "rb") as src, open(tmp, "wb") as dst:
        dst.write(hdr)
        src.seek(rd.data_offset)
        shutil.copyfileobj(src, dst, 1 << 26)
    data_start = len(hdr)
    with open(tmp, "r+b") as dst:
        for off, eb in patches:
            dst.seek(data_start + off)
            dst.write(eb)
    os.replace(tmp, out)
    print(f"wrote {out}: header {rd.data_offset} -> {data_start} bytes, {len(patches)} tensors replaced")


if __name__ == "__main__":
    main()
