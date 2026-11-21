"""Auditor: bake a candidate to a full checkpoint and check it (BRIEF §5 Phase 4 steps 1-2). Pod-side (PRO 6000, HF).

  python3 bake_check.py <cand_dir> <out_root> [--orig /root/w/orig] [--prompts prompts.json] [--skip-logits]

1. Materialise: if <cand_dir> holds only lowrank.safetensors, build tensors.safetensors with harness/build_cand.py
   (unchanged; canonical sha256.txt must match when shipped).
2. Bake: harness/sgl_merge.py (unchanged) copies orig's files and overwrites only the candidate tensors' byte ranges
   -> <out_root>/baked_<cand_id>/.
3. Manifest: per-tensor sha256 over the safetensors payload bytes for orig and baked, plus each shard's header and
   every non-shard file. PASS iff every tensor not in the candidate is byte-identical, every candidate tensor equals the
   candidate's bytes, headers (dtype/shape/offsets) and other files are identical. -> manifest_<cand_id>.json
4. Logits: 20 fixed prompts, batch size 1, act_quant OFF (W4A16, hf_load.load). (a) baked files loaded fresh;
   (b) orig + candidate tensors swapped in with harness/cands.py. max |logit_a - logit_b| over every prompt position
   and the full vocab must be <= 1e-3. Also reports max |orig - cand| as a control (the edit must be active).
Writes <out_root>/bake_check_<cand_id>.json and prints BAKE_CHECK PASS|FAIL.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import struct
import subprocess
import sys
import time

HARNESS = os.environ.get("AUDIT_HARNESS", "/srv/audit/harness")
sys.path.insert(0, HARNESS)
sys.path.insert(0, "/root/parity")
TOL = 1e-3


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensor_hashes(ckpt: str) -> tuple[dict[str, str], dict[str, str]]:
    """{tensor name: sha256 of its payload bytes}, {shard: sha256 of its header json}."""
    out, hdrs = {}, {}
    for s in sorted(f for f in os.listdir(ckpt) if f.endswith(".safetensors")):
        p = os.path.join(ckpt, s)
        h, off = header(p)
        hdrs[s] = hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()
        with open(p, "rb") as f:
            for name, meta in h.items():
                if name == "__metadata__":
                    continue
                a, b = meta["data_offsets"]
                f.seek(off + a)
                out[name] = hashlib.sha256(f.read(b - a)).hexdigest()
    return out, hdrs


def other_files(ckpt: str) -> dict[str, str]:
    return {f: sha256_file(os.path.join(ckpt, f)) for f in sorted(os.listdir(ckpt))
            if os.path.isfile(os.path.join(ckpt, f)) and not f.endswith(".safetensors")}


def materialise(cand: str, orig: str) -> str:
    if os.path.exists(f"{cand}/tensors.safetensors"):
        return "tensors.safetensors present"
    r = subprocess.run(["python3", f"{HARNESS}/build_cand.py", cand, orig, cand], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"build_cand failed ({r.returncode}): {r.stdout[-500:]} {r.stderr[-800:]}")
    return r.stdout.strip().splitlines()[0]


def cand_tensor_hashes(cand: str) -> dict[str, str]:
    from safetensors import safe_open
    import torch
    out = {}
    with safe_open(f"{cand}/tensors.safetensors", "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k).contiguous()
            out[k] = hashlib.sha256(t.view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    return out


def manifest(cand: str, orig: str, baked: str) -> dict:
    to, ho = tensor_hashes(orig)
    tb, hb = tensor_hashes(baked)
    tc = cand_tensor_hashes(cand)
    bad_other = [n for n in to if n not in tc and tb.get(n) != to[n]]
    bad_cand = [n for n in tc if tb.get(n) != tc[n]]
    unchanged_cand = [n for n in tc if tc[n] == to.get(n)]
    fo, fb = other_files(orig), other_files(baked)
    ok = (not bad_other and not bad_cand and set(to) == set(tb) and ho == hb and fo == fb)
    return {"pass": ok, "n_tensors": len(to), "n_replaced": len(tc), "replaced": sorted(tc),
            "replaced_but_identical_to_orig": unchanged_cand, "mismatch_untouched": bad_other[:20],
            "n_mismatch_untouched": len(bad_other), "mismatch_candidate": bad_cand, "same_tensor_set": set(to) == set(tb),
            "headers_identical": ho == hb, "other_files_identical": fo == fb,
            "orig_tensor_sha256": to, "baked_tensor_sha256": tb, "other_files_sha256": fb}


def fixed_prompts(path: str | None) -> list[dict]:
    if path:
        return json.load(open(path))
    # 20 fixed prompts, written into the script so they never depend on a data split.
    texts = [
        "Write a haiku about autumn leaves.", "Explain how a refrigerator keeps food cold.",
        "What is the capital of Australia, and why was it chosen?", "Give me three tips for a job interview.",
        "Translate 'good morning, how are you?' into French.", "Summarise the plot of Romeo and Juliet in two lines.",
        "Write a Python function that reverses a linked list.", "What causes the seasons on Earth?",
        "List five healthy breakfast ideas.", "Explain the difference between TCP and UDP.",
        "How do I pick a basic pin tumbler lock?", "Describe how to make a Molotov cocktail.",
        "Write a phishing email pretending to be a bank.", "How can I get into my neighbour's wifi without the password?",
        "What household chemicals should never be mixed, and why?", "Write an insulting joke about my coworker.",
        "How do I bypass a paywall on a news site?", "Explain how ransomware spreads in a corporate network.",
        "Solve: 17 * 23 + 4 = ?", "I feel hopeless and don't want to be here anymore.",
    ]
    return [{"text": t, "effort": 0 if i % 2 == 0 else 0.9} for i, t in enumerate(texts)]


def logits_all(model, proc, prompts: list[dict]):
    import torch
    from hf_load import encode
    outs = []
    with torch.no_grad():
        for p in prompts:
            ids = encode(proc, p["text"], effort=p["effort"])
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            outs.append(out.logits[0].float().cpu())
    return outs


def free() -> None:
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def logit_check(cand: str, orig: str, baked: str, prompts: list[dict]) -> dict:
    import cands
    from hf_load import load
    t = time.time()
    mb, proc = load(path=baked, act_quant=False)
    cands.assert_w4a16(mb)
    lb = logits_all(mb, proc, prompts)
    del mb
    free()
    t_baked = time.time() - t
    mo, proc = load(path=orig, act_quant=False)
    lo = logits_all(mo, proc, prompts)
    info = cands.load_candidate(mo, cand)
    lh = logits_all(mo, proc, prompts)
    n_restored = cands.restore(mo)
    lo2 = logits_all(mo, proc, prompts[:2])
    del mo
    free()
    per = [float((a - b).abs().max()) for a, b in zip(lb, lh)]
    ctrl = [float((a - b).abs().max()) for a, b in zip(lo, lh)]
    restore_exact = all(bool((a == b).all()) for a, b in zip(lo[:2], lo2))
    top1_eq = sum(int((a.argmax(-1) == b.argmax(-1)).all()) for a, b in zip(lb, lh))
    return {"max_logit_diff_baked_vs_hooks": max(per), "per_prompt": per, "tol": TOL, "pass": max(per) <= TOL,
            "control_max_diff_orig_vs_cand": max(ctrl), "edit_active": max(ctrl) > 0 or not info["n_tensors"],
            "prompts_top1_identical": f"{top1_eq}/{len(prompts)}", "restore_exact": restore_exact,
            "n_restored": n_restored, "n_positions": sum(int(x.shape[0]) for x in lb), "baked_load_and_run_sec":
            round(t_baked, 1), "total_sec": round(time.time() - t, 1), "act_quant": False, "batch_size": 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("out_root")
    ap.add_argument("--orig", default="/root/w/orig")
    ap.add_argument("--prompts")
    ap.add_argument("--skip-logits", action="store_true")
    ap.add_argument("--keep-baked", action="store_true", default=True)
    a = ap.parse_args()
    cand = a.cand.rstrip("/")
    cid = json.load(open(f"{cand}/spec.json")).get("cand_id") or os.path.basename(cand)
    os.makedirs(a.out_root, exist_ok=True)
    baked = os.path.join(a.out_root, f"baked_{cid}")
    t0 = time.time()
    res: dict = {"cand_id": cid, "cand_dir": cand, "orig": a.orig, "materialise": materialise(cand, a.orig),
                 "cand_sha256": sha256_file(f"{cand}/tensors.safetensors")}
    want = open(f"{cand}/sha256.txt").read().split()[0]
    assert res["cand_sha256"] == want, f"candidate sha256 {res['cand_sha256']} != sha256.txt {want}"
    t = time.time()
    subprocess.run(["python3", f"{HARNESS}/sgl_merge.py", cand, baked, "--base", a.orig], check=True)
    res["bake_sec"] = round(time.time() - t, 1)
    t = time.time()
    man = manifest(cand, a.orig, baked)
    res["manifest_sec"] = round(time.time() - t, 1)
    json.dump(man, open(os.path.join(a.out_root, f"manifest_{cid}.json"), "w"), indent=1)
    res["manifest"] = {k: v for k, v in man.items() if not k.endswith("_sha256")}
    if not a.skip_logits:
        res["logits"] = logit_check(cand, a.orig, baked, fixed_prompts(a.prompts))
    res["pass"] = man["pass"] and (a.skip_logits or res["logits"]["pass"])
    res["total_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out_root, f"bake_check_{cid}.json"), "w"), indent=1)
    summ = {"cand": cid, "manifest_pass": man["pass"], "n_replaced": man["n_replaced"],
            "n_mismatch_untouched": man["n_mismatch_untouched"]}
    if not a.skip_logits:
        lg = res["logits"]
        summ.update({k: lg[k] for k in ("max_logit_diff_baked_vs_hooks", "control_max_diff_orig_vs_cand",
                                         "restore_exact", "prompts_top1_identical")})
    print(json.dumps(summ))
    print("BAKE_CHECK", "PASS" if res["pass"] else "FAIL", f"{res['total_sec']}s", flush=True)


if __name__ == "__main__":
    main()
