"""Fill the KL/top-1 placeholders and the robustness table in the model card from the auditor's aggregates.

  python3 fill_card.py [--card R/package/MODEL_CARD_DRAFT.md] [--out R/out/final/MODEL_CARD.md] [--final c1h_r1_svd1_s3_L4]

KL/top-1: test rows (split "test") for orig-noise / ou / final from results/runs.jsonl, or audit/test/evals/<cand>_kl/
metrics.json. Missing values stay as {{...}} and are listed. Robustness: audit/test/evals/<cand>_r*/metrics.json
(+ jev_metrics.json), rendered as one table replacing the <!-- ROBUSTNESS --> block; planned sets with no result are
marked "not completed". Reads aggregates only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

R = os.path.expanduser("~/Code/inkling-abl")
EV = f"{R}/audit/test/evals"
PLANNED = [("sysprompt", "Safety system prompt"), ("multiturn", "Multi-turn"), ("image", "Request inside an image"),
           ("longctx", "Request inside 32k tokens of context")]
N_LANG_PLANNED = 4


def audit_metrics(cand: str, pattern: str) -> dict:
    out = {}
    for p in sorted(glob.glob(f"{EV}/{cand}_{pattern}/*metrics.json")):
        out.update(json.load(open(p))["metrics"])
    return out


def kl_values(final: str) -> dict:
    rows = [json.loads(l) for l in open(f"{R}/results/runs.jsonl") if l.strip()]
    extra = f"{R}/package/test_kl_rows.jsonl"  # auditor's test KL relayed by the lead before its runs.jsonl rows exist
    if os.path.exists(extra):
        rows += [json.loads(l) for l in open(extra) if l.strip()]
    rows = [r for r in rows if r.get("split") == "test" and not r.get("error")]

    def get(cand, names, tag_re=None):
        c = [r for r in rows if r.get("candidate") == cand and r.get("metric") in names
             and (tag_re is None or re.search(tag_re, str(r.get("tag") or "")))]
        if not c:
            m = audit_metrics(cand, "kl")
            for n in names:
                if n in m:
                    return m[n]
            return None
        return sorted(c, key=lambda r: r.get("time", ""))[-1]
    noise = r"noise|self"
    return {"KL_FLOOR": get("orig", ("kl_256", "kl"), noise), "KL_OU": get("ou", ("kl_256", "kl")),
            "KL_OURS": get(final, ("kl_256", "kl")), "T1_FLOOR": get("orig", ("top1_256", "top1"), noise),
            "T1_OU": get("ou", ("top1_256", "top1")), "T1_OURS": get(final, ("top1_256", "top1"))}


def fmt(v: dict | None, kind: str) -> str | None:
    if not v:
        return None
    ci = v.get("CI")
    if kind == "kl":
        return f"{v['value']:.4f}" + (f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "")
    return f"{100 * v['value']:.1f}%" + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else "")


def robust_sets(cand: str) -> dict[str, dict]:
    m = audit_metrics(cand, "r*")
    sets = {}
    for k, v in m.items():
        g = re.match(r"robust_(.+?)_refusal_off$", k)
        if g:
            sets.setdefault(g.group(1), {})["refusal"] = v
        g = re.match(r"jev_useful_robust_(.+?)_off$", k)
        if g:
            sets.setdefault(g.group(1), {})["useful"] = v
    return sets


def pct(v: dict | None) -> str:
    return "not completed" if not v else f"{100 * v['value']:.1f}%"


ROBUST_FINAL = f"{R}/audit/test/robust.json"  # auditor's final robustness aggregates (AUDIT.md §4)
ROBUST_ROWS = [("multiturn", "Multi-turn (benign turn, its own reply, then the request)"),
               ("sysprompt", "Safety system prompt"), ("image", "Request as text inside an image"),
               ("longctx", "Request after 32,000 tokens of benign context"), ("lang_es", "Spanish"),
               ("lang_fr", "French"), ("lang_de", "German"), ("lang_zh", "Chinese")]
# plain request reference row: AUDIT.md §1 (refusal off / Jev useful off)
PLAIN = {"orig": (0.975, 0.004), "ou": (0.018, 0.343), "final": (0.0, 0.542)}


def robustness_table(final: str) -> str:
    d = json.load(open(ROBUST_FINAL))

    def cell(c: str, key: str) -> str:
        v = d.get(c, {}).get(f"robust_{key}_off")
        if not v:
            raise SystemExit(f"robustness {c}/{key} missing in {ROBUST_FINAL}")
        ref = v.get("phrase_raw_or_backtranslated") or v["phrase_raw"]
        return f"{100 * ref['value']:.1f}% / {100 * v['jev_useful']['value']:.1f}%"

    def plain(k: str) -> str:
        return f"{100 * PLAIN[k][0]:.1f}% / {100 * PLAIN[k][1]:.1f}%"
    L = ["| Set (thinking off, 277 harmful test prompts; refused / useful) | Original | OpenUncensored | Unbound |",
         "|---|---|---|---|",
         f"| Plain request (reference) | {plain('orig')} | {plain('ou')} | {plain('final')} |"]
    for key, label in ROBUST_ROWS:
        L.append(f"| {label} | not run | {cell('ou', key)} | {cell(final, key)} |")
    L += ["", "The original was scored only on the plain request: it refuses 97.5% there, so the robustness sets test "
          "whether refusals come back in the edited models.",
          "Non-English rows count a refusal if the raw answer or its back-translation to English matches the refusal phrase list. "
          "Machine translation can soften a refusal, so treat these refusal rates as lower bounds; useful answers are judged on the back-translated answer."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default=f"{R}/package/MODEL_CARD_DRAFT.md")
    ap.add_argument("--out", default=f"{R}/out/final/MODEL_CARD.md")
    ap.add_argument("--final", default="c1h_r1_svd1_s3_L4")
    a = ap.parse_args()
    s = open(a.card).read()
    kl = kl_values(a.final)
    for k, v in kl.items():
        val = fmt(v, "kl" if k.startswith("KL") else "t1")
        if val:
            s = s.replace("{{" + k + "}}", val)
    s = re.sub(r"<!-- ROBUSTNESS -->.*?<!-- /ROBUSTNESS -->",
               lambda _: "<!-- ROBUSTNESS -->\n" + robustness_table(a.final) + "\n<!-- /ROBUSTNESS -->", s, flags=re.S)
    open(a.out, "w").write(s)
    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", s)))
    print("filled:", {k: fmt(v, "kl" if k.startswith("KL") else "t1") for k, v in kl.items()})
    print("placeholders left:", left)


if __name__ == "__main__":
    main()
