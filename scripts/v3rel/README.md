# v3rel: stage v3 privately, go live on the owner's word, build and verify the Basalt pack

Nothing here makes anything public until the go-live step, and go-live needs `--yes-owner-approved` plus the file
`~/.inkling-abl/v3_golive_approved`, which only the owner creates, on each machine that runs a go-live step.
v2.1 was staged on a branch of the public repo (`promote_v21.py`). Branches of a public repo are public, so v3 does
not stage that way.

| File | What it does |
|---|---|
| `stage_v3.py` | Creates the 4 private repos and uploads to them. Checks privacy before and after, verifies sha256, writes a receipt. `--dry-run` makes no network calls. `--selftest` |
| `verify_staged.py` | Read-only check of the 4 private repos: privacy, files, sha256 against the receipts, card lint, small-file diff. `--selftest` |
| `golive_v3.py` | `check` (read-only, writes the plan), then `prime-public`, `unbound-promote`, `gguf-upload`, `basalt-upload`, `tag`, `cleanup` and `rollback`. `--selftest` |
| `basalt_v3.sh` | Runs on a pod. Builds the Basalt pack from a merged checkpoint with Basalt v0.5.5, then runs the verifier |
| `verify_basalt_v3.py` | The v3 pack check: every byte must equal either the official pack or the converter's output from the merged checkpoint. `--selftest` |
| `v3rel_common.py` | Shared code: names, hashing, privacy guard, receipts, approval gate, and the fake hub used by the selftests |

Repos. The private allow-list holds only these four: `nokeepai/Inkling-2-Mini-Prime`, `-Prime-LoRA` and
`-Prime-GGUF` (new), and `nokeepai/Inkling-2-Mini-Unbound-v3-staging` (new, deleted after go-live). The existing
public repos are `nokeepai/Inkling-2-Mini-Unbound`, `nokeepai/Inkling-2-Mini-Unbound-GGUF` and
`nokeepai/Inkling-2-Mini-Unbound-Basalt`. All three have tags `v1.0`, `v2.0` and `v2.1`, plus branches `v2` and
`v2.1`. If Prime is renamed (owner rule: "Honed"), pass `--prime-name Honed` to every script.

Private storage estimate (about 45 GB allowed):

| Repo | Size |
|---|---|
| Prime | 20.2 GB: shard 19.53, mtp 0.64, tokenizer and configs |
| Prime-LoRA | 1.46 GB |
| Unbound-v3-staging | 19.5 GB: only the files that differ from public Unbound `main`, normally the shard, README and v3 charts |
| Total | about 41.2 GB |

`stage_v3.py` refuses to go past `--budget-gb` (default 45). It counts the receipts on the machine it runs on.

Token. The token is never passed, printed or stored. huggingface_hub finds it itself, as the v2.1 scripts did
(`HfApi()`, `get_token()`): env `HF_TOKEN`, else the token file at `$HF_TOKEN_PATH`, which defaults to
`$HF_HOME/token`. On pods, `HF_HOME=/workspace/hf` as in rel21.sh. On the Mac, point the hub at the write-token
file without reading it yourself: `export HF_TOKEN_PATH=~/.inkling-abl/hf_write_token`. Receipts and the plan file
contain no token, and the selftests check this with a sentinel token.

Python. On the Mac, use `PY=~/.inkling-abl/tinker_lab/.venv/bin/python` (huggingface_hub 2.4.0). Do not use
the Homebrew python3.13: it has 0.35.3, which has no cross-repo copy (see the hub API notes), and its install is
broken (`import huggingface_hub` fails because `requests` is missing). On a pod, use the pod's `python3`. All
three scripts print the hub version they run with.

## (a) Staging tonight

### 0. Offline checks on the Mac (no network)

```bash
cd <worktree>; PY=~/.inkling-abl/tinker_lab/.venv/bin/python
for s in stage_v3 golive_v3 verify_staged verify_basalt_v3; do $PY -I scripts/v3rel/$s.py --selftest | tail -1; done   # 4x SELFTEST_OK
bash -n scripts/v3rel/basalt_v3.sh
```

### 1. Final cards and charts

Follow `package_v3/README.md` steps 4 and 5 (`render.py --final`, `make_v3.py --final`). `stage_v3.py` refuses any
card that still contains `{{`, `TBD`, `DRAFT RENDER` or `[MISSING`.

**There is no Prime-GGUF card yet.** `render.py` has no template for it, so `prime-gguf-card-only` waits until
`package_v3/dist/Inkling-2-Mini-Prime-GGUF/README.md` exists. Alternatively, pass `--card F`.

### 2. From the Mac: the LoRA and all cards and charts

```bash
A=<final adapter dir>      # e.g. ~/.inkling-abl/tinker_lab/adapters/w18s60/<run id>  (adapter_model.safetensors, 1.46 GB)
M=scripts/v3stack/merge_adapter.py
$PY -I scripts/v3rel/stage_v3.py prime-lora --adapter "$A" --merge-script $M --dry-run
$PY -I scripts/v3rel/stage_v3.py prime-lora --adapter "$A" --merge-script $M
$PY -I scripts/v3rel/stage_v3.py prime-main      --only card      # README + hero/speed/nothing_lost_prime/indep_prime charts
$PY -I scripts/v3rel/stage_v3.py unbound-staging --only card      # Unbound README + all of package_v3/charts
$PY -I scripts/v3rel/stage_v3.py prime-gguf-card-only             # only once its card exists (see step 1)
```

Each run ends with `STAGE PASS <repo> @ <sha>`, and writes `~/.inkling-abl/v3rel_receipts/<repo>__<part>.json`.

### 3. On the pod that holds `/srv/stack_w/prime` and `/srv/stack_w/unbound`

`H=root@ip` and `P=port` come from the rental.

```bash
rsync -az -e "ssh -p $P" --exclude __pycache__ scripts/v3rel/ $H:/root/v3rel/
ssh -p $P $H
  test -s "$HF_HOME/token" && echo token-present        # never cat it
  python3 -c 'import huggingface_hub as h; print(h.__version__)'
  cd /root/v3rel; mkdir -p /root/logs
  python3 stage_v3.py prime-main      --only weights --ckpt /srv/stack_w/prime   --dry-run
  python3 stage_v3.py unbound-staging --only weights --ckpt /srv/stack_w/unbound --dry-run
  nohup sh -c 'python3 stage_v3.py prime-main --only weights --ckpt /srv/stack_w/prime;
               python3 stage_v3.py unbound-staging --only weights --ckpt /srv/stack_w/unbound' > /root/logs/stage_v3.log 2>&1 &
  grep -E 'PRIVATE OK|PLAN|COMMITTED|FAIL|STAGE|ABORT|identical to public' /root/logs/stage_v3.log
```

Other checkpoint paths can be passed with `--ckpt`. The run uploads the 12 stock-layout files only, never the
checkpoint dir's own README.md or merge report. On `unbound-staging` it skips every file identical to public
Unbound `main` (normally `mtp.safetensors`, the tokenizer and the configs), and its log prints which. A re-run
resumes: files already on the hub with the same sha256 are skipped. Expect about 25 to 60 minutes for the two
19.5 GB uploads, depending on uplink.

### 4. Bring the receipts together and verify (read-only)

```bash
rsync -az -e "ssh -p $P" $H:/root/.inkling-abl/v3rel_receipts/ ~/.inkling-abl/v3rel_receipts/
$PY -I scripts/v3rel/verify_staged.py                         # ends VERIFY_STAGED PASS
#   on the pod as well, to diff configs/tokenizer against the merged checkpoints (copy the Mac receipts there first):
#   rsync -az -e "ssh -p $P" ~/.inkling-abl/v3rel_receipts/ $H:/root/.inkling-abl/v3rel_receipts/
#   python3 /root/v3rel/verify_staged.py --local nokeepai/Inkling-2-Mini-Prime=/srv/stack_w/prime \
#       --local nokeepai/Inkling-2-Mini-Unbound-v3-staging=/srv/stack_w/unbound
```

### 5. Build the Basalt pack and the GGUF ladders on the pod (no upload)

The GGUF ladders come from `scripts/v3stack/gguf_v3.sh`. For Basalt:

```bash
nohup bash /root/v3rel/basalt_v3.sh /srv/stack_w/unbound /srv/basalt_v3 \
    Inkling-2-Mini-Unbound-v3-NVFP4-q8.basalt "Inkling 2 Mini Unbound v3" [/srv/stock] > /root/logs/basalt_v3.out 2>&1 &
grep -E 'START|CONVERT|SHA|VERIFY|STOP|ALL_DONE' /srv/basalt_v3/basalt_v3.log
```

Put the pack's size, sha256 and a one-line summary of `basalt_verify.log` into `values.json` (`files.basalt_*`),
then re-render.

## (b) Go-live, only on the owner's word

Run `check` first. It is read-only, prints every change per repo (adds, replacements, copies, proposed deletions,
tags) and saves the plan to `~/.inkling-abl/v3_golive_plan.json`. Every mutating step recomputes its own part of
the plan and refuses if anything differs from what `check` printed. Re-run `check` before a step whenever an
earlier step changed a repo it reads (`tag` and `cleanup` read all of them). A step that ran is logged in
`~/.inkling-abl/v3_golive_plan.applied.json`, and `rollback` reads that log.

```bash
# owner, on each machine that runs a go-live step:  touch ~/.inkling-abl/v3_golive_approved
G="$PY -I scripts/v3rel/golive_v3.py"
$G check --propose-delete 'charts/._*'          # read every line; the AppleDouble junk on Unbound main is only PROPOSED
$G prime-public    --yes-owner-approved          # Prime, Prime-LoRA, Prime-GGUF -> public (re-verified vs receipts first)
$G unbound-promote --yes-owner-approved          # ONE commit on Unbound main: server-side copies from the staging repo
#    add --with-deletions only if the owner wants the proposed deletions applied
$G check && $G tag --yes-owner-approved          # v3.0 on Unbound (GGUF/Basalt get it after their uploads; re-run)
```

From the pod. Copy the receipts and the plan there, or run `check` there too.

```bash
G="python3 /root/v3rel/golive_v3.py"
awk '{print $2"\t"$1}' /srv/gguf_v3/unbound/SHA256SUMS | while IFS=$'\t' read f s; do
  printf '%s\t%s\t%s\n' "$f" "$(stat -c %s /srv/gguf_v3/unbound/$f)" "$s"; done > /root/v3_manifest_rows.tsv
$G check --gguf-dir /srv/gguf_v3/unbound --manifest-rows /root/v3_manifest_rows.tsv --card-unbound-gguf <card> \
         --prime-gguf-dir /srv/gguf_v3/prime --card-prime-gguf <card> \
         --basalt-pack /srv/basalt_v3/Inkling-2-Mini-Unbound-v3-NVFP4-q8.basalt --card-basalt <card> \
         --verify-log /srv/basalt_v3/basalt_verify.log
$G gguf-upload --yes-owner-approved --which unbound --gguf-dir /srv/gguf_v3/unbound --manifest-rows /root/v3_manifest_rows.tsv --card <card>
$G gguf-upload --yes-owner-approved --which prime   --gguf-dir /srv/gguf_v3/prime --card <card>
$G basalt-upload --yes-owner-approved --basalt-pack /srv/basalt_v3/Inkling-2-Mini-Unbound-v3-NVFP4-q8.basalt \
    --card <card> --verify-log /srv/basalt_v3/basalt_verify.log
$G check && $G tag --yes-owner-approved          # v3.0 on GGUF and Basalt (Unbound's existing tag is left alone)
```

Last step, from either machine:

```bash
$G check && $G cleanup                                   # prints; refuses while any staged file is missing from Unbound main
$G cleanup --yes-owner-approved --delete-staging         # deletes nokeepai/Inkling-2-Mini-Unbound-v3-staging
```

What each step guarantees:

- **prime-public:** refuses unless all three repos match their receipts exactly and the cards pass the lint.
- **unbound-promote:** the commit pins `parent_commit`. Files that exist only on main (`lora_v2.1/`, `finetune_v2.1/`,
  `finetune/`, `lora_v2/`, `abliteration/`, `lora/`, older charts) are kept, and the step checks afterwards that they
  are unchanged. Nothing is deleted unless `check` listed it and `--with-deletions` is given.
- **gguf-upload and basalt-upload:** they refuse a repo that is still private (storage cap). They upload one commit
  per large file, so a failed run resumes. Only `README.md`, `MANIFEST.tsv` and `SHA256SUMS` may be overwritten, and
  nothing is ever deleted. GGUF files are cross-checked against the build's `SHA256SUMS`. Basalt also needs the
  verifier's `RESULT: VERIFY PASS` and a matching `PACK_SHA256` line.
- **tag:** refuses to tag Unbound unless main carries the staged v3 shard.

## (c) Rollback, one per go-live step

Every rollback needs the same gate (`--yes-owner-approved` plus the approval file). It undoes only what the
applied log says that step changed.

| Step | Rollback | What it does |
|---|---|---|
| prime-public | `$G rollback --yes-owner-approved --step prime-public` | Flips the three Prime repos back to private. Anyone who downloaded in the meantime keeps the files. |
| unbound-promote | `$G rollback --yes-owner-approved --step unbound-promote` | One commit on main. Every path the promotion touched goes back to its state at the pre-promotion commit (an intra-repo server-side copy for LFS files, re-added for small ones). Paths that v3 added are deleted. The applied deletions are restored. The v3 commit stays in history. Also roll back `tag`. |
| tag | `$G rollback --yes-owner-approved --step tag [--tag v3.0]` | Deletes only the tags this script created. |
| gguf-upload | `$G rollback --yes-owner-approved --step gguf-unbound` (or `gguf-prime`) | Deletes the v3 GGUF files and restores README/MANIFEST to the pre-upload commit. |
| basalt-upload | `$G rollback --yes-owner-approved --step basalt` | Deletes the v3 pack and restores the README. |
| cleanup | none | The staging repo is gone, so re-stage with `stage_v3.py unbound-staging` if needed. That is why cleanup runs last and checks first that main carries every staged file. |

Manual fallback if the scripts cannot run: v2.1 stays reachable at tag `v2.1` (and branch `v2.1`) on all three
Unbound repos. To restore Unbound main by hand, use `promote_v21.py`'s method, a server-side `CommitOperationCopy`
with `src_revision="v2.1"`.

## Hub API notes (read before go-live)

- **Visibility flip.** Both installed versions have `HfApi.update_repo_settings(repo_id, private=False)`: 2.4.0 in
  the tinker venv and 0.35.3 in Homebrew python3.13. 2.4.0 also accepts `visibility=` (`public`, `private`,
  `protected`). The scripts use `private=` and re-read `repo_info().private` afterwards.
- **Cross-repo server-side copy.**
  - huggingface_hub 2.4.0 has `CommitOperationCopy(..., src_repo_id=, src_repo_type=)`. Its docstring says LFS
    objects are duplicated to the destination before the commit, same storage region only.
  - 0.35.3 has no `src_repo_id`, so it can only copy within one repo. `golive_v3.py` detects which one it has.
  - **UNVERIFIED on the real hub:** whether a copy from a PRIVATE source into a PUBLIC destination works, and whether
    it works for Xet-backed files. Not tested, because no hub calls were made.
  - Fallback without 2.x: `unbound-promote --from-local /srv/stack_w/unbound --card F --charts D` on the pod
    re-uploads, after checking each local sha256 against the staged one.
- **LFS sha256.** Verification reads `model_info(files_metadata=True).siblings[].lfs.sha256`. If the hub ever omits
  it for a large (Xet) file, the scripts report `no LFS sha256 on hub` as a FAIL. They do not assume a match.
- **Repo creation.** `create_repo(private=True, exist_ok=True)` leaves an existing repo's visibility as it is. That
  is why `stage_v3.py` re-reads privacy and aborts on anything but `private is True`, both before and after the
  commit.

## Basalt: what was found

- **Converter.** Basalt v0.5.5 = commit `ff0b978d2ee9ff195d113b6feb80961b2d2f99ef`. Run
  `tools/convert/convert.py --checkpoint DIR`. The inkling2 path is `checkpoint_inkling2.py`.
  - NVFP4 experts (layers 0-19): E2M1 codes and E4M3 scales are copied as stored, repacked into 64-value blocks,
    plus an FP32 tail `amax / outer_scale_denominator`.
  - MXFP8 experts (layers 20-23): ggml reference Q8_0 of their exact values.
  - BF16 attention projections, `attn_rel` (= `wr_du`) included: Q8_0.
  - Head, embedding and loop controllers: BF16. Norms, router and the rest: F32.
  - Vision tower and DSpark drafter: as stored.
  - So v3's re-quantised expert codes go into the pack bit for bit, and the build works unchanged.
- **Official reference pack.** It must be the 0.5.5 pack, sha `f7b4e5f8…3787`: `Inkling-2-Mini-NVFP4-DQ8.basalt`
  (formerly `-q8`) at revision `6c9fbf0730a6d2e8813a1b262c2ab3548cf0239e`. On 2026-11-23 it was removed from
  jesdga/Inkling-2-Mini-Basalt `main`, and main now has only the 0.5.6 `-MX8` pack (native MXFP8 layers, a different
  layout). `scripts/v21/rel21.sh` downloads the DQ8 file from `main` and would fail today. `basalt_v3.sh` pins the
  revision.
- **Open (owner's call).** v3 keeps the 0.5.5 `-q8` recipe, which matches the repo's v1.0/v2/v2.1 files and the
  card's filename. A 0.5.6 MX8 build would need a new verifier and a new card.
- **Not covered by the v3 verifier:** engine-level behaviour (`basalt serve` sanity, refusal and crisis probes).
  That needs a Blackwell GPU and is not part of this path.
