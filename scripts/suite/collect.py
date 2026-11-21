"""Mac (agent `suite`): pull score.json files from ink-suite-1, append runs.jsonl rows once per (model, bench),
write suite/RESULTS.md.   python3 collect.py [--no-fetch]"""
from __future__ import annotations

import json
import os
import subprocess
import sys

R = os.path.expanduser("~/Code/inkling-abl")
POD = "ink-suite-1"
LOCAL = f"{R}/suite/out"
APPENDED = f"{R}/suite/appended.json"
MODELS = ["orig", "ou"]
BENCH_ORDER = ["mmlu_pro", "gsm8k", "math500", "gpqa_diamond", "ifeval", "humaneval_plus", "speed", "speed_dspark",
               "ruler"]
GPU = {"ruler": "RTXPRO6000"}


def ink(*a: str) -> str:
    return subprocess.run(["bash", f"{R}/scripts/ink_lium.sh", *a], capture_output=True, text=True).stdout


def fetch() -> None:
    os.makedirs(LOCAL, exist_ok=True)
    out = ink("exec", POD, "cd /srv/suite/out && tar czf /tmp/scores.tgz $(find . -name score.json -not -path './smoke/*')"
              " && echo OK")
    if "OK" not in out:
        print("no scores yet")
        return
    subprocess.run(["bash", f"{R}/scripts/ink_lium.sh", "scp", POD, "/tmp/scores.tgz", f"{LOCAL}/scores.tgz",
                    "--download"], capture_output=True, text=True)
    subprocess.run(["tar", "xzf", f"{LOCAL}/scores.tgz", "-C", LOCAL], check=True)


def scores() -> dict:
    res: dict = {}
    for m in sorted(os.listdir(LOCAL)):
        d = os.path.join(LOCAL, m)
        if not os.path.isdir(d):
            continue
        for b in os.listdir(d):
            p = os.path.join(d, b, "score.json")
            if os.path.exists(p):
                res.setdefault(m, {})[b] = json.load(open(p))
    return res


def append_rows(res: dict) -> None:
    done = set(json.load(open(APPENDED))) if os.path.exists(APPENDED) else set()
    rows = []
    for m, bs in res.items():
        for b, s in bs.items():
            tag = f"{m}|{b}"
            if tag in done:
                continue
            for name, v in s["metrics"].items():
                rows.append({"run_id": f"suite-{m}-{b}", "agent": "suite", "split": "suite", "metric": name,
                             "candidate": m, "value": v["value"], "CI": v["CI"], "n": v["n"],
                             "pod": "ink-suite-lc" if b == "ruler" else POD, "gpu": GPU.get(b, "RTX5090"),
                             "engine": "sglang_frozen" if b != "speed_dspark" else "sglang_frozen+dspark",
                             "main": name == s["main"]})
            done.add(tag)
    if rows:
        p = subprocess.run([sys.executable, f"{R}/scripts/append_run.py"], input="\n".join(json.dumps(r) for r in rows),
                           text=True, capture_output=True)
        assert p.returncode == 0, p.stderr
        json.dump(sorted(done), open(APPENDED, "w"), indent=0)
        print(f"appended {len(rows)} rows")


def fmt(v: dict | None) -> str:
    if not v:
        return "–"
    ci = v.get("CI")
    return f"{v['value']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{v['value']:.1f}"


def results_md(res: dict) -> None:
    fins = [f["name"] for f in json.load(open(f"{R}/suite/finalists.json"))]
    models = [m for m in MODELS + fins if m in res] + [m for m in res if m not in MODELS + fins]
    benches = [b for b in BENCH_ORDER if any(b in res[m] for m in models)]
    L = ["# Phase 4 capability suite: results (agent `suite`)", "",
         "Settings and scope: suite/PLAN.md. Versions: suite/VERSIONS.md. 95% bootstrap CIs over items (10,000 "
         "resamples, seed 1234). Every benchmark ran on one GPU type for every model (RTX 5090, frozen ENGINE.md engine).",
         "Verdict: finalist point estimate inside orig's 95% CI → 'inside'; outside → 'below'/'above'.", "",
         "## Main metrics", "", "| Benchmark | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for b in benches:
        mains = {m: res[m][b]["metrics"][res[m][b]["main"]] for m in models if b in res[m]}
        L.append(f"| {b} | " + " | ".join(fmt(mains.get(m)) for m in models) + " |")
    L += ["", "## Verdict per finalist (inside orig's 95% CI?)", "", "| Benchmark | " + " | ".join(fins) + " |",
          "|---|" + "---|" * len(fins)]
    for b in benches:
        if b.startswith("speed") or "orig" not in res or b not in res["orig"]:
            continue
        o = res["orig"][b]["metrics"][res["orig"][b]["main"]]
        cells = []
        for f in fins:
            if f not in res or b not in res[f]:
                cells.append("pending")
                continue
            v = res[f][b]["metrics"][res[f][b]["main"]]["value"]
            cells.append("inside" if o["CI"][0] <= v <= o["CI"][1] else ("below" if v < o["CI"][0] else "above")
                         + f" ({v - o['value']:+.2f})")
        L.append(f"| {b} | " + " | ".join(cells) + " |")
    L += ["", "## Thinking hygiene and secondary metrics", ""]
    for b in benches:
        names = sorted({k for m in models if b in res[m] for k in res[m][b]["metrics"]} - {res[models[0]][b]["main"]
                                                                                           if b in res[models[0]] else ""})
        if not names:
            continue
        L += [f"### {b}", "", "| Metric | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for k in names:
            L.append(f"| {k} | " + " | ".join(fmt(res[m][b]["metrics"].get(k)) if b in res[m] else "–"
                                                for m in models) + " |")
        L.append("")
    open(f"{R}/suite/RESULTS.md", "w").write("\n".join(L) + "\n")
    print("wrote suite/RESULTS.md")


if __name__ == "__main__":
    if "--no-fetch" not in sys.argv:
        fetch()
    r = scores()
    append_rows(r)
    results_md(r)
