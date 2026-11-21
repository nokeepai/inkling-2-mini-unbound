"""Auditor: sealed-split SGLang scoring for one candidate (BRIEF §5 Phase 4 step 3). Pod-side, frozen engine.

  python3 sealed_run.py --cand <id|orig> --data <dir with <split>/{harmful,borderline,probes}.jsonl> --split test \
      [--gpu i | --port P] [--blocks core,cap,robust] [--robust <dir from robust_build.py>] [--think-cap 4096]
      [--tag audit] [--limit N] [--efforts 0,0.9]

Scoring code is harness/sgl_eval.py and harness/fleet_eval.py, imported UNCHANGED: S.capped / S.eval_generative /
S.is_refusal (phrase list) / S.CRISIS / S.LIMIT / S.hygiene / S.stat (bootstrap 10k, seed 1234) / S.swap / S.restore,
F.prepare (sha / compact build) and F.ensure_orig (pinned orig probe baseline per container). Only the inputs differ:
the split directory and the final-run thinking cap (4,096; answer cap 512).

Blocks
  core:   harmful refusal at effort 0 ("off") and 0.9 (+ per category, hygiene: think_len, finish, cap_hit, rep);
          full borderline over-refusal (effort 0); probes at effort 0 and 0.9 (honest_limit, crisis).
  cap:    MMLU-300 (letter logprobs) and GSM8K-100 (effort 0.5), as the inner loop (sanity only; the verdict comes
          from the full capability suite).
  robust: items from robust_build.py: multiturn, sysprompt, longctx, lang_{es,fr,de,zh} through S.capped with the
          official template (messages rendered by the tokenizer's chat template); image through the server's
          OpenAI endpoint (multimodal path; reasoning parser splits thinking; max_tokens = think cap + 512).
Outputs: /srv/audit/evals/<cand>_<tag>/{gens_*.jsonl, metrics.json, done.json | error.json}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.request

HARNESS = os.environ.get("AUDIT_HARNESS", "/srv/audit/harness")
sys.path.insert(0, HARNESS)
import sgl_eval as S  # noqa: E402
import fleet_eval as F  # noqa: E402

ANS_CAP = 512


def render_msgs(messages: list[dict], effort: float) -> list[int]:
    s = S.TOK.apply_chat_template(messages, tokenize=False, reasoning_effort=effort)
    return S.TOK(s, add_special_tokens=False).input_ids


def harmful_block(harm, efforts, cap, out_dir, met, timing) -> None:
    for eff in efforts:
        name = "off" if eff == 0 else f"e{str(eff).replace('0.', '0')}"
        t = time.time()
        res = S.eval_generative(harm, eff, cap, ANS_CAP, f"{out_dir}/gens_harmful_{name}.jsonl")
        met[f"refusal_{name}"] = S.stat([float(r["refusal"]) for r in res])
        for c in sorted({r["category"] for r in res}):
            met[f"refusal_{name}_cat_{c}"] = S.stat([float(r["refusal"]) for r in res if r["category"] == c])
        met.update(S.hygiene(res, f"harmful_{name}"))
        timing[f"harmful_{name}"] = round(time.time() - t, 1)


def core(a, out_dir, met, timing, lim, parts=("harm", "border", "probes")) -> None:
    if "harm" in parts:
        harmful_block(lim(S.load("harmful", a.split)), a.efforts, a.think_cap, out_dir, met, timing)
    if "border" in parts:
        border(a, out_dir, met, timing, lim)
    if "probes" in parts:
        probes(a, out_dir, met, timing)


def border(a, out_dir, met, timing, lim) -> None:
    t = time.time()
    res = S.eval_generative(lim(S.load("borderline", a.split)), 0, a.think_cap, ANS_CAP, f"{out_dir}/gens_borderline.jsonl")
    met["overrefusal_off"] = S.stat([float(r["refusal"]) for r in res])
    for c in sorted({r["category"] for r in res}):
        met[f"overrefusal_off_cat_{c}"] = S.stat([float(r["refusal"]) for r in res if r["category"] == c])
    met.update(S.hygiene(res, "borderline_off"))
    timing["borderline"] = round(time.time() - t, 1)


def probes(a, out_dir, met, timing) -> None:
    for eff in a.efforts:
        name = "off" if eff == 0 else f"e{str(eff).replace('0.', '0')}"
        t = time.time()
        res = S.eval_generative(S.load("probes", a.split), eff, a.think_cap, ANS_CAP, f"{out_dir}/gens_probes_{name}.jsonl")
        met[f"probe_crisis_resources_{name}"] = S.stat([float(bool(S.CRISIS.search(S.norm(r["answer"])))) for r in res
                                                       if r["category"] == "crisis"])
        met[f"probe_honest_limit_{name}"] = S.stat([float(bool(S.LIMIT.search(S.norm(r["answer"])))) for r in res
                                                   if r["category"] == "honest_limit"])
        timing[f"probes_{name}"] = round(time.time() - t, 1)


def cap_block(out_dir, met, timing) -> None:
    t = time.time()
    met.update(S.mmlu(out_dir))
    timing["mmlu"] = round(time.time() - t, 1)
    t = time.time()
    met.update(S.gsm8k(out_dir))
    timing["gsm8k"] = round(time.time() - t, 1)


def chat_image(it: dict, effort: float, cap: int) -> dict:
    body = {"model": "default", "temperature": 0, "max_tokens": cap + ANS_CAP, "separate_reasoning": True,
            "chat_template_kwargs": {"reasoning_effort": effort},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + it["image_b64"]}},
                {"type": "text", "text": it["text"]}]}]}
    req = urllib.request.Request(f"http://localhost:{S.PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    o = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    ch = o["choices"][0]
    ans = ch["message"].get("content") or ""
    think = ch["message"].get("reasoning_content") or ""
    think_ids = S.TOK(think, add_special_tokens=False).input_ids if think else []
    all_ids = think_ids + S.TOK(ans, add_special_tokens=False).input_ids
    return {"answer": ans, "think_len": len(think_ids), "cap_hit": len(think_ids) >= cap and not ans,
            "finished": ch.get("finish_reason") == "stop", "rep8x4": S.repetition(all_ids),
            "loop": S.loop_detect(all_ids), "think_tail": think[-200:], "usage": o.get("usage")}


def robust_one(it: dict, effort: float, cap: int) -> dict:
    kind = it["kind"]
    if kind == "image":
        r = chat_image(it, effort, cap)
    else:
        msgs = list(it["messages"])
        if kind == "multiturn":
            first = S.capped(render_msgs(msgs[:1], effort), cap, ANS_CAP)
            msgs = [msgs[0], {"role": "assistant", "content": first["answer"]}, msgs[-1]]
            it = {**it, "turn1_answer": first["answer"][:2000]}
        r = S.capped(render_msgs(msgs, effort), cap, ANS_CAP)
        r.pop("answer_ids")
    r.update({"id": it["id"], "kind": kind, "category": it.get("category"), "prompt": it["prompt"], "effort": effort,
              "lang": it.get("lang", "en"), "refusal": int(S.is_refusal(r["answer"]))})
    if "turn1_answer" in it:
        r["turn1_answer"] = it["turn1_answer"]
    return r


def robust(a, out_dir, met, timing, lim) -> None:
    kinds = sorted(f[:-6] for f in os.listdir(a.robust) if f.endswith(".jsonl"))
    for kind in kinds:
        items = lim([json.loads(l) for l in open(f"{a.robust}/{kind}.jsonl")])
        for eff in a.robust_efforts:
            name = "off" if eff == 0 else f"e{str(eff).replace('0.', '0')}"
            t = time.time()
            res = S.run_pool(lambda it: robust_one(it, eff, a.think_cap), items)
            with open(f"{out_dir}/gens_robust_{kind}_{name}.jsonl", "w") as f:
                for r in res:
                    f.write(json.dumps(r) + "\n")
            met[f"robust_{kind}_refusal_{name}"] = S.stat([float(r["refusal"]) for r in res])
            met.update(S.hygiene(res, f"robust_{kind}_{name}"))
            timing[f"robust_{kind}_{name}"] = round(time.time() - t, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--blocks", default="core,cap,robust")
    ap.add_argument("--robust", default="")
    ap.add_argument("--think-cap", type=int, default=4096)
    ap.add_argument("--efforts", default="0,0.9")
    ap.add_argument("--robust-efforts", default="0")
    ap.add_argument("--tag", default="audit")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.efforts = [float(x) if "." in x else int(x) for x in a.efforts.split(",")]
    a.robust_efforts = [float(x) if "." in x else int(x) for x in a.robust_efforts.split(",")]
    S.DATA = a.data
    if a.gpu is not None:
        F.CONTAINER, F.SLOT = f"sgl{a.gpu}", f"__g{a.gpu}"
    S.PORT = a.port or F.gpu_port(a.gpu)
    out_dir = f"/srv/audit/evals/{a.cand}_{a.tag}"
    os.makedirs(out_dir, exist_ok=True)
    for f in ("done.json", "error.json"):
        if os.path.exists(f"{out_dir}/{f}"):
            os.remove(f"{out_dir}/{f}")
    lim = (lambda xs: xs[:a.limit]) if a.limit else (lambda xs: xs)
    blocks = a.blocks.split(",")
    t0 = time.time()
    met: dict = {}
    timing: dict = {}
    sw, prep = None, None
    try:
        base = F.ensure_orig()
        if a.cand != "orig":
            prep = F.prepare(a.cand)
            sw = S.swap(a.cand + F.SLOT)
            assert sw["base"] == base, "pre-swap probe differs from the pinned orig baseline"
            timing["swap"] = sw["swap_sec"]
        parts = ("harm", "border", "probes") if "core" in blocks else tuple(b for b in blocks if b in ("harm", "border", "probes"))
        if parts:
            core(a, out_dir, met, timing, lim, parts)
        if "cap" in blocks:
            cap_block(out_dir, met, timing)
        if "robust" in blocks and a.robust:
            robust(a, out_dir, met, timing, lim)
        if sw:
            t = time.time()
            S.restore(base)
            timing["restore"] = round(time.time() - t, 1)
            sw = None
        summary = {"cand": a.cand, "tag": a.tag, "split": a.split, "data": a.data, "think_cap": a.think_cap,
                   "answer_cap": ANS_CAP, "efforts": a.efforts, "port": S.PORT, "container": F.CONTAINER,
                   "metrics": met, "timing": timing, "prep": prep, "limit": a.limit,
                   "total_sec": round(time.time() - t0, 1)}
        json.dump(summary, open(f"{out_dir}/metrics.json", "w"), indent=1)
        json.dump({"ok": True, "total_sec": summary["total_sec"]}, open(f"{out_dir}/done.json", "w"))
        print(json.dumps({k: round(v["value"], 4) for k, v in met.items() if "_cat_" not in k}), timing, flush=True)
    except Exception as e:  # noqa: BLE001 - record and restore orig
        err = {"error": f"{type(e).__name__}: {e}"[:1500], "trace": traceback.format_exc()[-3000:], "partial": met}
        try:
            if sw:
                S.restore(sw["base"])
            else:
                F.ensure_orig()
            err["restored"] = True
        except Exception as e2:  # noqa: BLE001
            err["restored"], err["restore_error"] = False, f"{type(e2).__name__}: {e2}"[:800]
        json.dump(err, open(f"{out_dir}/error.json", "w"), indent=1)
        print(json.dumps(err)[:3000], file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        import shutil
        shutil.rmtree(f"/srv/w/merged_{a.cand}{F.SLOT}", ignore_errors=True)


if __name__ == "__main__":
    main()
