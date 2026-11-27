"""Shared pieces for the v3 release scripts (stage_v3.py, golive_v3.py, verify_staged.py).

Repo names, the stock checkpoint layout, local hashing (sha256 for LFS files, git blob sha1 for small files), the
remote file listing, the privacy guard, receipts, and a FakeApi used only by the offline --selftest modes.

Token: never handled here. huggingface_hub's own lookup is used, exactly as the v2.1 scripts do (HfApi() with no
token argument): env HF_TOKEN first, else the token file ($HF_HOME/token, ~/.cache/huggingface/token). We only ask
whether one exists (get_token() is not None) and never print, log or store it.
"""
from __future__ import annotations

import datetime as _dt
import fnmatch
import hashlib
import inspect
import json
import os
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ORG = "nokeepai"
HOME_STATE = Path.home() / ".inkling-abl"
RECEIPTS_DIR = HOME_STATE / "v3rel_receipts"
SHA_CACHE = HOME_STATE / "v3rel_sha_cache.json"
APPROVAL_FILE = HOME_STATE / "v3_golive_approved"
PLAN_FILE = HOME_STATE / "v3_golive_plan.json"
GB = 1e9

# The stock repo's layout (thinkingmachines/Inkling-2-Mini @ ed40e7da, same list as scripts/v21/rel21.sh and
# scripts/v11b/ckpt_basalt.sh). README.md is NOT taken from the checkpoint dir: merge_adapter.py hard-links the base's
# README (stock or v2.1 card) into its output, so the rendered card is always passed separately.
STOCK_LAYOUT = ("chat_template.jinja", "config.json", "generation_config.json", "hf_quant_config.json",
                "model-00001-of-00001.safetensors", "model.safetensors.index.json", "mtp.safetensors",
                "processor_config.json", "special_tokens_map.json", "tiktoken/tokenizer.model", "tokenizer.json",
                "tokenizer_config.json")
LARGE_FILES = ("model-00001-of-00001.safetensors", "mtp.safetensors")
PRIME_CHARTS = ("hero_v3", "speed_v3", "nothing_lost_prime_v3", "indep_prime_v3", "xver_v3")  # package_v3/README.md table
GGUF_TYPES_UNBOUND = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ3_M", "IQ2_M")       # gguf_README.md table (no BF16)
GGUF_TYPES_PRIME = ("BF16",) + GGUF_TYPES_UNBOUND                                 # owner brief: BF16 too
UNBOUND_BASALT_FILE = "Inkling-2-Mini-Unbound-v3-NVFP4-q8.basalt"                  # basalt_README.md


@dataclass(frozen=True)
class Names:
    prime: str
    prime_lora: str
    prime_gguf: str
    staging: str
    unbound: str = f"{ORG}/Inkling-2-Mini-Unbound"
    unbound_gguf: str = f"{ORG}/Inkling-2-Mini-Unbound-GGUF"
    unbound_basalt: str = f"{ORG}/Inkling-2-Mini-Unbound-Basalt"

    @property
    def private_allow(self) -> tuple[str, ...]:
        """The only repos stage_v3.py may ever write to."""
        return (self.prime, self.prime_lora, self.prime_gguf, self.staging)


def names(prime_name: str = "Prime") -> Names:
    """prime_name follows package_v3/values.json names.prime_name (owner rule: may become "Honed")."""
    if not prime_name.isalnum():
        raise SystemExit(f"bad --prime-name {prime_name!r}")
    p = f"{ORG}/Inkling-2-Mini-{prime_name}"
    return Names(prime=p, prime_lora=p + "-LoRA", prime_gguf=p + "-GGUF",
                 staging=f"{ORG}/Inkling-2-Mini-Unbound-v3-staging")


# ---------------------------------------------------------------- hub access
def hub_version() -> str:
    try:
        import huggingface_hub
        return huggingface_hub.__version__
    except ImportError:
        return "not installed"


def real_api() -> Any:
    from huggingface_hub import HfApi
    return HfApi()  # token: HF_TOKEN env or the standard token file, never passed or printed here


def require_token() -> None:
    from huggingface_hub import get_token
    if not get_token():
        raise SystemExit("no HF token: set HF_TOKEN or log in (token file). It is never printed.")


def supports_cross_repo_copy() -> bool:
    """CommitOperationCopy(src_repo_id=...) exists in huggingface_hub 2.x (2.4.0 checked), not in 0.35.x."""
    try:
        from huggingface_hub import CommitOperationCopy
    except ImportError:
        return False
    return "src_repo_id" in inspect.signature(CommitOperationCopy).parameters


def is_private(api: Any, repo: str) -> bool | None:
    return getattr(api.repo_info(repo), "private", None)


def assert_private(api: Any, repo: str, allow: tuple[str, ...]) -> None:
    """Abort unless repo is in the allow-list and the hub says private is True (None or False both abort)."""
    if repo not in allow:
        raise SystemExit(f"ABORT: {repo} is not in the private allow-list {list(allow)}")
    p = is_private(api, repo)
    if p is not True:
        raise SystemExit(f"ABORT: {repo} reports private={p!r}; nothing is uploaded to a repo that is not private")
    print(f"PRIVATE OK {repo}", flush=True)


def ensure_private_repo(api: Any, repo: str, allow: tuple[str, ...]) -> None:
    """Create as private if missing (exist_ok keeps an existing repo's visibility as is), then re-read and assert."""
    if repo not in allow:
        raise SystemExit(f"ABORT: {repo} is not in the private allow-list {list(allow)}")
    api.create_repo(repo, private=True, repo_type="model", exist_ok=True)
    assert_private(api, repo, allow)


def _lfs_sha(sib: Any) -> str | None:
    lfs = getattr(sib, "lfs", None)
    if lfs is None:
        return None
    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)


def remote_files(api: Any, repo: str, revision: str = "main") -> tuple[str, dict[str, dict]]:
    """-> (commit sha, {path: {size, sha256 (LFS) or None, blob_id}}) from model_info(files_metadata=True)."""
    info = api.model_info(repo, revision=revision, files_metadata=True)
    out = {}
    for s in info.siblings or []:
        out[s.rfilename] = {"size": s.size, "sha256": _lfs_sha(s), "blob_id": getattr(s, "blob_id", None)}
    return info.sha, out


# ---------------------------------------------------------------- local hashing
def _load_cache() -> dict:
    try:
        return json.loads(SHA_CACHE.read_text())
    except (OSError, ValueError):
        return {}


def sha256_file(path: Path, cache: bool = True) -> str:
    """sha256 of a file, cached by (realpath, size, mtime_ns): a 19.5 GB shard is hashed once per change."""
    st = path.stat()
    key = f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime_ns}"
    c = _load_cache() if cache else {}
    if key in c:
        return c[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    d = h.hexdigest()
    if cache:
        c = _load_cache()
        c[key] = d
        SHA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SHA_CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(c, indent=0))
        tmp.replace(SHA_CACHE)
    return d


def git_blob_sha1(path: Path) -> str:
    """The hub's blob_id for a non-LFS file: git's sha1 of b"blob <len>\\0" + content."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def local_meta(path: Path, hash_it: bool = True) -> dict:
    size = path.stat().st_size
    m: dict[str, Any] = {"size": size}
    if hash_it:
        m["sha256"] = sha256_file(path)
        if size < 50 << 20:  # small enough to be a plain git file on the hub
            m["blob_id"] = git_blob_sha1(path)
    return m


def same_content(local: dict, remote: dict | None) -> bool:
    """True only when the remote copy is provably the same bytes (LFS sha256, or git blob id for small files)."""
    if not remote or remote.get("size") != local["size"]:
        return False
    if remote.get("sha256"):
        return remote["sha256"] == local.get("sha256")
    return bool(remote.get("blob_id")) and remote["blob_id"] == local.get("blob_id")


def verify_against(expect: dict[str, dict], remote: dict[str, dict], exact: bool,
                   ignore: tuple[str, ...] = (".gitattributes",)) -> list[str]:
    """Every expected file present with equal size and content id; exact=True also forbids unexpected remote files."""
    fails = []
    for p, m in sorted(expect.items()):
        r = remote.get(p)
        if r is None:
            fails.append(f"missing on hub: {p}")
        elif r.get("size") != m["size"]:
            fails.append(f"size differs: {p} hub {r.get('size')} local {m['size']}")
        elif r.get("sha256"):
            if r["sha256"] != m.get("sha256"):
                fails.append(f"sha256 differs: {p}")
        elif m["size"] >= 10 << 20:
            fails.append(f"no LFS sha256 on hub for large file {p} (cannot verify; see README 'hub API notes')")
        elif r.get("blob_id") != m.get("blob_id"):
            fails.append(f"git blob id differs: {p}")
    if exact:
        extra = sorted(set(remote) - set(expect) - set(ignore))
        fails += [f"unexpected file on hub: {p}" for p in extra]
    return fails


# ---------------------------------------------------------------- receipts, approval
def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def receipt_path(repo: str, part: str, root: Path = RECEIPTS_DIR) -> Path:
    return root / f"{repo.replace('/', '__')}__{part}.json"


def write_receipt(path: Path, data: dict) -> None:
    tok = os.environ.get("HF_TOKEN")
    text = json.dumps(data, indent=1, sort_keys=True)
    if tok and tok in text:  # cannot happen (we never read it); belt and braces
        raise SystemExit("ABORT: receipt would contain the token")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    print(f"RECEIPT {path}", flush=True)


def load_receipts(repo: str, root: Path = RECEIPTS_DIR) -> dict[str, dict]:
    """All parts' files for one repo, merged (a later receipt for the same path wins)."""
    files: dict[str, dict] = {}
    for p in sorted(root.glob(f"{repo.replace('/', '__')}__*.json"), key=lambda q: json.loads(q.read_text())["timestamp"]):
        files.update(json.loads(p.read_text())["files"])
    return files


def staged_bytes(root: Path = RECEIPTS_DIR, exclude: str = "") -> float:
    tot = 0
    for p in root.glob("*.json"):
        r = json.loads(p.read_text())
        if r.get("repo") != exclude and r.get("private") is True:
            tot += sum(m["size"] for m in r["files"].values())
    return tot


def approval_gate(flag: bool, approval_file: Path = APPROVAL_FILE) -> None:
    if not flag:
        raise SystemExit("REFUSED: mutating go-live step needs --yes-owner-approved")
    if not approval_file.is_file():
        raise SystemExit(f"REFUSED: {approval_file} does not exist (the owner creates it, on this machine)")
    print(f"APPROVAL OK ({approval_file})", flush=True)


def match_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def host() -> str:
    return socket.gethostname()


# ---------------------------------------------------------------- selftest double
@dataclass
class FakeRepo:
    private: bool
    files: dict[str, dict] = field(default_factory=dict)  # path -> {size, sha256, blob_id}
    commits: int = 0
    tags: list[str] = field(default_factory=list)
    history: dict[str, dict] = field(default_factory=dict)  # commit sha -> files at that commit
    tagmap: dict[str, str] = field(default_factory=dict)    # tag -> commit sha


class FakeApi:
    """In-memory stand-in for HfApi, for the offline selftests only. Records every mutating call."""

    def __init__(self, repos: dict[str, FakeRepo] | None = None, corrupt: Callable[[str], bool] | None = None):
        self.repos = repos or {}
        self.calls: list[tuple] = []
        self.corrupt = corrupt or (lambda p: False)

    def _sha(self, repo: str) -> str:
        return hashlib.sha1(f"{repo}:{self.repos[repo].commits}".encode()).hexdigest()

    def create_repo(self, repo, private=None, repo_type="model", exist_ok=False):
        self.calls.append(("create_repo", repo, private))
        if repo not in self.repos:
            self.repos[repo] = FakeRepo(private=bool(private))
        elif not exist_ok:
            raise RuntimeError("exists")

    def _rev(self, repo, revision):
        r = self.repos[repo]
        return r.tagmap.get(revision, revision)

    def snapshot(self, repo):
        """Record the current state under the current sha (a fake repo seeded with files has no commit yet)."""
        r = self.repos[repo]
        r.history[self._sha(repo)] = {k: dict(v) for k, v in r.files.items()}

    def repo_info(self, repo, **_):
        r = self.repos[repo]
        return type("I", (), {"private": r.private, "sha": self._sha(repo)})()

    def model_info(self, repo, revision="main", files_metadata=False):
        r = self.repos[repo]
        files = r.files if revision in (None, "main", self._sha(repo)) else r.history[self._rev(repo, revision)]
        sibs = [type("S", (), {"rfilename": p, "size": m["size"], "blob_id": m.get("blob_id"),
                               "lfs": ({"sha256": m["sha256"]} if m["size"] >= 10 << 20 else None)})()
                for p, m in files.items()]
        return type("M", (), {"siblings": sibs, "sha": self._sha(repo), "private": r.private})()

    def create_commit(self, repo, operations, commit_message="", parent_commit=None, **_):
        self.calls.append(("create_commit", repo, [type(o).__name__ + ":" + o.path_in_repo for o in operations]))
        r = self.repos[repo]
        if parent_commit and parent_commit != self._sha(repo):
            raise RuntimeError("parent moved")
        for o in operations:
            kind = type(o).__name__
            if kind == "CommitOperationAdd":
                p = Path(o.path_or_fileobj)
                m = local_meta(p)
                if self.corrupt(o.path_in_repo):
                    m["sha256"] = "0" * 64
                    m["blob_id"] = "0" * 40
                m["_src"] = str(p)
                r.files[o.path_in_repo] = m
            elif kind == "CommitOperationDelete":
                r.files.pop(o.path_in_repo, None)
            elif kind == "CommitOperationCopy":
                srepo = getattr(o, "src_repo_id", None) or repo
                src = self.repos[srepo]
                rev = getattr(o, "src_revision", None)
                sfiles = src.files if rev in (None, self._sha(srepo)) else src.history[self._rev(srepo, rev)]
                r.files[o.path_in_repo] = dict(sfiles[o.src_path_in_repo])
        r.commits += 1
        r.history[self._sha(repo)] = {k: dict(v) for k, v in r.files.items()}
        return type("C", (), {"oid": self._sha(repo), "commit_url": f"fake://{repo}/{self._sha(repo)}"})()

    def update_repo_settings(self, repo, private=None, **_):
        self.calls.append(("update_repo_settings", repo, private))
        self.repos[repo].private = bool(private)

    def create_tag(self, repo, tag, revision=None, tag_message=None, exist_ok=False):
        self.calls.append(("create_tag", repo, tag))
        self.repos[repo].tags.append(tag)
        self.repos[repo].tagmap[tag] = revision or self._sha(repo)

    def hf_hub_download(self, repo_id, filename, revision=None, **_):
        """Small files only; the selftest keeps every revision's content for the paths it reverts."""
        r = self.repos[repo_id]
        hist = getattr(r, "history", {})
        rev = self._rev(repo_id, revision) if revision else None
        m = hist[rev][filename] if rev in hist else r.files[filename]
        return m["_src"]

    def delete_tag(self, repo, tag, **_):
        self.calls.append(("delete_tag", repo, tag))
        self.repos[repo].tags.remove(tag)

    def delete_repo(self, repo, **_):
        self.calls.append(("delete_repo", repo))
        del self.repos[repo]

    def list_repo_refs(self, repo, **_):
        tags = [type("T", (), {"name": t})() for t in self.repos[repo].tags]
        return type("R", (), {"tags": tags, "branches": []})()


def die_on_token_leak(text: str) -> None:
    tok = os.environ.get("HF_TOKEN")
    if tok and tok in text:
        print("SELFTEST FAIL: token sentinel leaked into output", file=sys.stderr)
        raise SystemExit(1)
