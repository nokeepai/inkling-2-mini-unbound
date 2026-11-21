"""Pod job: (1) ou and c0 candidate folders, c0 == ou bit-exact; (2) hf_name_map.json by value matching;
(3) load/restore round trip of c0 through cands.py.
Outputs: /root/fid/cands/{ou,c0}/, /root/fid/hf_name_map.json, /root/fid/build_report.json
"""
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/root/parity")
sys.path.insert(0, "/root/fid")
ORIG_ST = "/root/w/orig/model-00001-of-00001.safetensors"
OU_ST = "/root/w/ou/model-00001-of-00001.safetensors"
OUT = "/root/fid"
REPORT: dict = {}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def bits(t):
    return t.contiguous().view(torch.uint8)


def write_cand(cid, tensors, params):
    d = f"{OUT}/cands/{cid}"
    os.makedirs(d, exist_ok=True)
    save_file(tensors, f"{d}/tensors.safetensors")
    spec = {"cand_id": cid, "family": "baselines", "parent": None,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "agent": "fidelity",
            "params": params, "kind": "weights"}
    json.dump(spec, open(f"{d}/spec.json", "w"), indent=1)
    open(f"{d}/sha256.txt", "w").write(sha(f"{d}/tensors.safetensors") + "  tensors.safetensors\n")


def build_baselines():
    t0 = time.time()
    with safe_open(ORIG_ST, "pt") as fo, safe_open(OU_ST, "pt") as fu:
        names = sorted(fo.keys())
        assert names == sorted(fu.keys())
        diff = [n for n in names if not torch.equal(bits(fo.get_tensor(n)), bits(fu.get_tensor(n)))]
        REPORT["ou_differing"] = diff
        assert len(diff) == 20 and all(re.fullmatch(r"model\.llm\.layers\.\d+\.attn\.wo_ud\.weight", n) for n in diff), diff
        ou = {n: fu.get_tensor(n).clone() for n in diff}
        write_cand("ou", ou, {"source": "OpenUncensored/Inkling-2-Mini-Uncensored@4e5f06d7", "tensors": len(ou)})
        edit = torch.load("/root/w/ou/edit.pt", map_location="cpu", weights_only=True)
        Q = edit["Q"].float()
        s_attn = edit["s_attn"].float()
        assert float(edit["s_mlp"].abs().sum()) == 0.0
        c0 = {}
        for i in range(24):
            s = float(s_attn[i])
            if s == 0.0:
                continue
            n = f"model.llm.layers.{i}.attn.wo_ud.weight"
            W32 = fo.get_tensor(n).float()
            c0[n] = (W32 - s * (Q @ (Q.T @ W32))).to(torch.bfloat16)
        assert sorted(c0) == sorted(ou), (sorted(c0), sorted(ou))
        mism = {n: int((bits(c0[n]) != bits(ou[n])).sum()) for n in c0}
        REPORT["c0_vs_ou_differing_bytes"] = mism
        assert all(v == 0 for v in mism.values()), mism
        write_cand("c0", c0, {"source": "edit.pt applied by fidelity: W32 - s*(Q@(Q.T@W32)) -> bf16",
                              "s_attn": [float(x) for x in s_attn]})
    REPORT["baselines_sec"] = round(time.time() - t0, 1)
    print("baselines ok: c0 == ou bit-exact on", len(c0), "tensors", flush=True)


@torch.no_grad()
def build_map(model):
    params = dict(model.named_parameters())
    by_numel: dict[int, list[str]] = {}
    for n, p in params.items():
        by_numel.setdefault(p.numel(), []).append(n)
    nm, unmatched, ambiguous = {}, [], {}
    with safe_open(ORIG_ST, "pt") as fo:
        keys = [k for k in fo.keys() if k.startswith("model.llm.")
                and not re.search(r"experts|w13_weight|w2_weight", k)]
        for k in sorted(keys):
            t = fo.get_tensor(k)
            m = re.search(r"\.layers\.(\d+)\.", k)
            cands = [n for n in by_numel.get(t.numel(), []) if "visual" not in n and "vision" not in n]
            if m:
                cands = [n for n in cands if f".layers.{m.group(1)}." in n]
            tf = t.float().flatten().cuda()
            hits = []
            for n in cands:
                p = params[n].detach().float().flatten()
                if torch.equal(p, tf):
                    hits.append((n, "none"))
                elif t.dim() == 2 and torch.equal(params[n].detach().float().reshape(t.shape[::-1]).T.flatten(), tf):
                    hits.append((n, "T"))
            if len(hits) > 1:  # break ties by the obvious name correspondence (wo_ud -> o_proj.linear etc.)
                ambiguous[k] = [h[0] for h in hits]
                hits = [h for h in hits if _name_hint(k, h[0])] or hits[:0]
            if len(hits) != 1:
                unmatched.append(k)
                continue
            n, tr = hits[0]
            p = params[n]
            nm[k] = {"hf": n, "transform": tr, "ckpt_dtype": str(t.dtype).replace("torch.", ""),
                     "hf_dtype": str(p.dtype).replace("torch.", ""), "ckpt_shape": list(t.shape),
                     "hf_shape": list(p.shape)}
    REPORT["map_matched"] = len(nm)
    REPORT["map_unmatched"] = unmatched
    REPORT["map_ambiguous"] = ambiguous
    json.dump(nm, open(f"{OUT}/hf_name_map.json", "w"), indent=1, sort_keys=True)
    return nm


HINTS = {"wo_ud": "o_proj.linear", "gate_attn": "o_proj.gate_proj", "attn_sconv": "post_attention_convolution",
         "mlp_sconv": "post_mlp_convolution", "attn_norm": "input_layernorm", "mlp_norm": "post_attention_layernorm",
         "mlp.gate.weight": "mlp.gate", "wq_du": "q_proj", "wk_dv": "k_proj", "wv_dv": "v_proj",
         "q_input_norm": "q_input_layernorm", "k_input_norm": "k_input_layernorm",
         "v_input_norm": "v_input_layernorm", "k_sconv": "k_proj", "v_sconv": "v_proj",
         "controllers.attn": "q_message_gate", "controllers.k": "k_message_gate", "controllers.v": "v_message_gate",
         "controllers.mlp": "mlp_message_gate", "norm.weight": "norm"}


def _name_hint(ck, hf):
    return any(a in ck and b in hf for a, b in HINTS.items())


def main():
    build_baselines()
    from hf_load import load
    t0 = time.time()
    model, _ = load(act_quant=False)
    REPORT["load_sec"] = round(time.time() - t0, 1)
    nm = build_map(model)
    need = [k for k in nm]
    REPORT["map_counts"] = {kind: sum(kind in k for k in need) for kind in
                            ("wo_ud", "attn_sconv", "mlp_sconv", "norm", "mlp.gate", "controllers")}
    os.environ["HF_NAME_MAP"] = f"{OUT}/hf_name_map.json"
    import cands
    cands._name_map = None
    cands.NAME_MAP_PATH = f"{OUT}/hf_name_map.json"
    t0 = time.time()
    info = cands.load_candidate(model, f"{OUT}/cands/c0")
    REPORT["c0_load_sec"] = round(time.time() - t0, 2)
    t0 = time.time()
    n = cands.restore(model)
    REPORT["c0_restore_sec"] = round(time.time() - t0, 2)
    REPORT["c0_roundtrip"] = {"loaded": info["n_tensors"], "restored": n}
    json.dump(REPORT, open(f"{OUT}/build_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in REPORT.items() if k not in ("map_ambiguous",)}, indent=1)[:4000])
    print("BUILD_DONE")


if __name__ == "__main__":
    main()
