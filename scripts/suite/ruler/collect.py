"""Local: pull score.json + timing.json for one model from ink-suite-lc and append runs.jsonl rows.
  python3 collect.py <model> [--no-append]
"""
import json
import os
import subprocess
import sys

R = os.path.expanduser("~/Code/inkling-abl")
POD = "ink-suite-lc"


def pull(model: str, fname: str) -> str:
    d = f"{R}/suite/out/{model}/ruler"
    os.makedirs(d, exist_ok=True)
    subprocess.run(["bash", f"{R}/scripts/ink_lium.sh", "scp", POD, f"/srv/suite/out/{model}/ruler/{fname}", d + "/",
                    "-d"], check=True, capture_output=True)
    return f"{d}/{fname}"


def main() -> None:
    model = sys.argv[1]
    s = json.load(open(pull(model, "score.json")))
    t = json.load(open(pull(model, "timing.json")))
    pull(model, "gens_0of1.jsonl")
    m = s["metrics"]
    print(model, {k: round(m[k]["value"], 2) for k in ("ruler_32k", "ruler_128k") if k in m}, "time", t)
    if "--no-append" in sys.argv:
        return
    for metric in ("ruler_32k", "ruler_128k"):
        row = {"run_id": f"suite-ruler-{model}", "agent": "suite", "split": "suite", "metric": metric,
               "candidate": model, "value": round(m[metric]["value"], 3),
               "CI": [round(x, 3) for x in m[metric]["CI"]], "n": m[metric]["n"], "pod": POD, "gpu": "RTXPRO6000"}
        subprocess.run(["python3", f"{R}/scripts/append_run.py", json.dumps(row)], check=True)
        print("appended", row)


if __name__ == "__main__":
    main()
