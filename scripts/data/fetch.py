"""Download every source pinned to a revision. Writes raw files to R/data/raw/ and
R/data/raw/manifest.json (source, url, revision, licence, local files).
Prints only file names and counts, never row text."""
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

R = Path("/Users/nikshepsvn/Code/inkling-abl")
RAW = R / "sealed" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
api = HfApi()

HF = {
    "mlabonne_harmful": ("mlabonne/harmful_behaviors", None),
    "mlabonne_harmless": ("mlabonne/harmless_alpaca", None),
    "alpaca": ("tatsu-lab/alpaca", None),
    "orbench": ("bench-llm/or-bench", "or-bench-hard-1k"),
    "sorrybench": ("sorry-bench/sorry-bench-202503", "question.jsonl"),
    "mmlu": ("cais/mmlu", "all/test"),
    "gsm8k": ("openai/gsm8k", "main/test"),
}
GH = {
    "advbench": ("llm-attacks/llm-attacks", ["data/advbench/harmful_behaviors.csv"]),
    "harmbench": ("centerforaisafety/HarmBench", ["data/behavior_datasets/harmbench_behaviors_text_all.csv"]),
    "strongreject": ("alexandrasouly/strongreject", ["strongreject_dataset/strongreject_dataset.csv"]),
    "xstest": ("paul-rottger/exaggerated-safety", ["xstest_prompts.csv"]),
}


def gh_json(path):
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return json.loads(out.stdout)


def fetch_hf(key, repo, prefix, manifest):
    try:
        info = api.dataset_info(repo, files_metadata=False)
    except Exception as e:
        manifest[key] = {"repo": repo, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(key, "ERROR", manifest[key]["error"])
        return
    sha = info.sha
    lic = None
    if info.card_data is not None:
        lic = info.card_data.get("license")
    files = [s.rfilename for s in info.siblings]
    if prefix:
        want = [f for f in files if f.startswith(prefix) or ("/" + prefix) in f]
    else:
        want = [f for f in files if f.endswith((".parquet", ".csv", ".jsonl", ".json")) and "README" not in f]
    local = []
    errs = []
    for f in want:
        try:
            p = hf_hub_download(repo, f, repo_type="dataset", revision=sha, local_dir=str(RAW / key))
            local.append(str(Path(p).relative_to(RAW)))
        except Exception as e:
            errs.append(f"{f}: {type(e).__name__}: {str(e)[:200]}")
    manifest[key] = {"repo": repo, "url": f"https://huggingface.co/datasets/{repo}", "revision": sha,
                     "license": lic, "files": local, "all_files": files[:200], "errors": errs}
    print(key, sha, lic, len(local), "files", len(errs), "errors")


def fetch_gh(key, repo, paths, manifest):
    try:
        meta = gh_json(f"repos/{repo}")
        branch = meta["default_branch"]
        sha = gh_json(f"repos/{repo}/commits/{branch}")["sha"]
        lic = (meta.get("license") or {}).get("spdx_id")
    except Exception as e:
        manifest[key] = {"repo": repo, "error": str(e)[:300]}
        print(key, "ERROR", manifest[key]["error"])
        return
    local = []
    errs = []
    for p in paths:
        url = f"https://raw.githubusercontent.com/{repo}/{sha}/{p}"
        dest = RAW / key / Path(p).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["curl", "-sSfL", "-o", str(dest), url], capture_output=True, text=True)
        if r.returncode == 0:
            local.append(str(dest.relative_to(RAW)))
        else:
            errs.append(f"{url}: {r.stderr.strip()[:200]}")
    manifest[key] = {"repo": repo, "url": f"https://github.com/{repo}", "revision": sha, "license": lic,
                     "files": local, "raw_urls": [f"https://raw.githubusercontent.com/{repo}/{sha}/{p}" for p in paths],
                     "errors": errs}
    print(key, sha, lic, len(local), "files", len(errs), "errors")


def main():
    only = set(sys.argv[1:])
    mpath = RAW / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    for k, (repo, prefix) in HF.items():
        if not only or k in only:
            fetch_hf(k, repo, prefix, manifest)
    for k, (repo, paths) in GH.items():
        if not only or k in only:
            fetch_gh(k, repo, paths, manifest)
    mpath.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
