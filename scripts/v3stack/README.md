# v3stack: does the stock-trained anti-stall adapter stack on Unbound v2.1?

The adapter `w11_off_step30` was trained with RL on the stock model (harmless prompts only). It took hard debugging
from 80% to 100% solved with near-zero no-code turns. This test adds it to Unbound v2.1 (attention-only edit, experts
identical to stock) and checks two things: v2.1 still answers what stock refuses (ubench), and it gains the debugging
fix (agent_hard_noforce) with no loss on main, hardmath, code and rp.

| File | What it does |
|---|---|
| `merge_adapter.py` | base + scale * alpha/r * B @ A into a full checkpoint dir. `--parts attn` or `attn_experts`. `--selftest` |
| `selftest_merge.py` | the offline self-test (synthetic tensors with the adapter's real name and shape pattern) |
| `stack_test.sh` | on the pod: fetch v2.1, build the variants, serve each, run the suites (generation only) |

## Variants

| Variant | Weights | Served by |
|---|---|---|
| `v21` | Unbound v2.1 (`nokeepai/Inkling-2-Mini-Unbound`, revision `v2.1`), untouched: the reference | frozen SGLang image |
| `v21_attn` | v2.1 + the adapter's 144 attention LoRAs merged (BF16) | frozen SGLang image |
| `v21_full` | v2.1 + attention + the 72 expert LoRAs, experts requantised (NVFP4 layers 0-19, MXFP8 20-23) | frozen SGLang image |

The frozen SGLang image is `lmsysorg/sglang@sha256:421aed3a…` (harness/ENGINE.md, state/pins.json), with the
ENGINE.md command and flags and only the model dir changed. All three variants are ordinary checkpoints, so none
needs runtime adapter loading. `SERVER=vllm` switches to `vllm/vllm-openai:inkling-2-mini`, the image that served
v2.1 publicly (scripts/v21/serve_run.sh). That path is UNVERIFIED on a single 32 GB card and its tag is not pinned.

**What `v21_full` can and cannot show.** The stored experts are already on the 4-bit grid. Requantising W + delta on
the same block scales snaps any element whose change is under half a grid step back to the old value. Most of the
NVFP4 expert delta will probably vanish. The merge prints the delta survival per layer before anything is served,
where 0 means `v21_full` ≈ `v21_attn` on that layer. Layers 20-23 are 8-bit, so more of their delta survives. To test
the full adapter with its expert part intact you need runtime LoRA. The fallback for that is at the end of this file.

## GPU

- **Proven path (SGLang):** an **RTX 5090 32 GB** is the smallest card that works. It is the frozen engine's card
  (ENGINE.md: `--tp 1`, `--mem-fraction-static 0.73815`, about 81k KV tokens, 31 running requests). Its
  `--attention-backend fa4` and the NVFP4 expert kernels were run only on sm120 (5090, RTX PRO 6000). An RTX PRO 6000
  96 GB also works and gives more KV room for the 48k-token agent transcripts. Use one GPU type for all three
  variants, because ENGINE.md found that GPU types disagree on greedy prefixes.
- **Fallback (runtime LoRA, custom vLLM from scripts/v3rl):** the v3rl plan tested a **1x B200** (CUDA 13 driver
  r580+, 250 GB free disk). With BF16 experts (about 52 GB) an 80 GB card is the theoretical floor. That floor is
  UNVERIFIED.

## Disk

About 60 GB under `/srv/stack_w`: v2.1 is 20.2 GB, and each merged variant adds one 19.5 GB shard copy, with the
other files hard-linked. The SGLang image takes another 25-40 GB under `/var/lib/docker`. Plan for 120 GB free.
The preflight needs 70 GB under `/srv/stack_w`. Weights go under `/srv`, not `/root`, because the pod's docker cannot
bind-mount the `/root` FUSE mount (ENGINE.md).

## Commands, in order

Run these from the repo root on the Mac. `H` is the pod's `root@ip` and `P` its ssh port, both from the lead's
rental.

```bash
# 1. local checks (no GPU, no network)
bash -n scripts/v3stack/stack_test.sh
~/.inkling-abl/tinker_lab/.venv/bin/python scripts/v3stack/merge_adapter.py --selftest     # ends SELFTEST_OK

# 2. copy scripts, adapter, eval data and the benchmark questions/tools to the pod (~1.5 GB, mostly the adapter)
H=root@<ip>; P=<port>; S="ssh -p $P"
A=~/.inkling-abl/tinker_lab/adapters/w11_off_step30/6ac7f10f-4d81-5443-95e4-29c42389e0de:train:0_weights_w11_off-step30
$S $H 'mkdir -p /srv/stack/scripts/tinker /srv/stack/adapter /srv/stack/eval_data /srv/stack/uncensoredbench/questions /root/logs'
rsync -az -e "$S" --exclude __pycache__ scripts/v3stack scripts/evalsuite $H:/srv/stack/scripts/
rsync -az -e "$S" --exclude __pycache__ scripts/tinker/agent_hard_tasks $H:/srv/stack/scripts/tinker/
rsync -az -e "$S" "$A/adapter_config.json" "$A/adapter_model.safetensors" $H:/srv/stack/adapter/
rsync -az -e "$S" ~/.inkling-abl/tinker_lab/{prompts_eval,prompts_hardmath,prompts_code,rp_eval,sysrules_tasks}.jsonl \
    $H:/srv/stack/eval_data/
rsync -az -e "$S" --exclude __pycache__ --exclude '*.html' ~/Code/uncensoredbench/tools ~/Code/uncensoredbench/prompts \
    $H:/srv/stack/uncensoredbench/
rsync -az -e "$S" --exclude screen_canaries.jsonl ~/Code/uncensoredbench/questions/*.jsonl \
    $H:/srv/stack/uncensoredbench/questions/

# 3. run on the pod. HF_TOKEN is optional (the repo is public). If it is needed, type it at the prompt so it stays
#    out of shell history. Do not put it in a file.
$S $H
  read -rs HF_TOKEN; export HF_TOKEN            # optional; press Enter for none
  nohup bash /srv/stack/scripts/v3stack/stack_test.sh > /root/logs/stack_test.log 2>&1 &
  grep -E ' (PASS|FAIL|SKIP) ' /root/logs/stack_test.log      # one line per stage; rerun the same nohup line to resume
  # knobs: VARIANTS="v21 v21_attn"  PASS2="main hardmath"  FULL_MIN_SURVIVAL=0.05  GPU=0  SERVER=vllm

# 4. pull results back (generations, merge reports, logs)
mkdir -p ~/.inkling-abl/stack_evals
rsync -az -e "$S" --exclude _smoke $H:/root/stack_evals/ ~/.inkling-abl/stack_evals/
rsync -az -e "$S" "$H:/srv/stack_w/*.merge_report.json" "$H:/srv/stack_w/v21.sha256" ~/.inkling-abl/stack_evals/
rsync -az -e "$S" "$H:/root/logs/stack_*" ~/.inkling-abl/stack_evals/logs/

# 5. judge locally (paid: needs OPENROUTER_API_KEY; ubench Bison ~660 calls ≈ $0.20, lab judges capped at $1 + $2)
cd scripts/evalsuite
for v in v21 v21_attn v21_full; do
  python3 run_suite.py --base-url http://unused/v1 --model x --tag $v --out-root ~/.inkling-abl/stack_evals \
      --suites ubench --judge-only --judges bison --ubench-conditions command
  python3 run_suite.py --base-url http://unused/v1 --model x --tag $v --out-root ~/.inkling-abl/stack_evals \
      --suites main,hardmath,rp --judge-only --judges bison,haiku
done

# 6. compare with v21 as the reference (":1", ":2" set the order; the names have no step number)
python3 select_ckpt.py --out-root ~/.inkling-abl/stack_evals --ref ~/.inkling-abl/stack_evals/v21 \
    ~/.inkling-abl/stack_evals/v21_attn:1 ~/.inkling-abl/stack_evals/v21_full:2 --conditions command
python3 compare.py v21 v21_attn v21_full --out-root ~/.inkling-abl/stack_evals \
    --suites agent_hard_noforce,ubench,main,hardmath,code,rp          # summary tables
```

How to read step 6. `select_ckpt.py` flags rows where a variant is significantly worse than v21. For the stall check
you want the opposite: compare `agent_hard_noforce` solve_rate and no_code_rate directly (also in
`/root/stack_evals/stack_table.tsv`). ubench runs 5 items per category x level cell, so per-cell rows are
underpowered. Read the overall and headline willingness rows, and treat per-cell flags as hints.

## Settings (same for every variant)

Effort 0.9, temperature 0.6, top_p 1.0, seed 1234, 1 repeat. Thinking cap 4,096, answer cap 8,192. 31 concurrent
requests, matching the engine's cap. ubench uses the `command` condition with `--limit 220` (5 per cell over 11
categories x 4 levels). The script checks that `reasoning_effort` reaches the template by comparing `prompt_tokens`
across efforts. It then runs the suite with `--allow-effort-unverified`, because run_suite.py's own probe needs a
vLLM `/tokenize` or llama.cpp `/apply-template` route, and it is UNVERIFIED whether SGLang has one. Each variant also
gets a 2-prompt client smoke test first: thinking must come back split, otherwise set `EVAL_EXTRA=--raw-markers`.

## Round 2: stochastic rounding, and stock

The first pod run showed that round-to-nearest throws most of the adapter away. The stored base weights sit exactly
on their storage grid, so any change under half a grid step rounds back to the old value. Expert delta survival was
0.002, and only about 23% of BF16 attention elements changed (rel_delta about 0.001, against a BF16 step of about
0.4-0.8%). `--rounding stochastic` rounds each element of W + delta to one of its two neighbouring stored values,
with probability proportional to proximity, so the expected stored value is W + delta. Elements whose delta is
exactly zero stay byte-identical. Each merge measures both rounding modes and writes the one requested. The report
`<variant>.merge_report.json` then holds, per group (`attn`, `attn.<module>`, `experts.nvfp4`, `experts.mxfp8`,
`experts.layerN`, `all`):

- `change_ratio`: |realised| / |intended|
- `cosine` and `projected_survival`: the scale of the realised change along the delta
- `spectral`: a sample (attention layers 0/11/23; expert layers 0/10/19/21, experts 0 and 77) of |error·v| / |delta·v|
  for the delta's top-32 right-singular vectors (`ratio_top`) and for random v (`ratio_rand`)

`ratio_top` well below 1 means the adapter's low-rank signal stands above the rounding noise.

What each variant isolates:

| Variant | What it isolates |
|---|---|
| `v21_attn` vs `v21` | The adapter's attention part, as nearest rounding stores it. |
| `v21_attn_sr` vs `v21_attn` | The attention part with the rounding loss removed in expectation. |
| `v21_full_sr` vs `v21_attn_sr` | The expert part, which nearest rounding erased almost entirely. |
| `stock` | The untouched stock checkpoint (`thinkingmachines/Inkling-2-Mini@ed40e7da`, as scripts/pod_fetch.sh pins it), served by the same engine. Its agent_hard_noforce result should reproduce the training service's 80% solved / 88% no-code baseline. |
| `stock_full_sr` | The full adapter on its own training base, through the same merge and rounding path. It should approach the service's 100% / ~0 result. |

How to read the two stock variants against `v21_full_sr`:
- `stock_full_sr` good but `v21_full_sr` not: the fix does not transfer to v2.1.
- Both short of the service's numbers: the loss is in merging and storage (rounding, serving), not in v2.1.
- `stock` itself far from 80% / 88%: the engine or the suite differs from the training service's sampler, and the
  comparison needs that offset.

Commands (on the pod, same nohup form as step 3; `PASS2=""` = pass 1 only, agent_hard_noforce then ubench):

```bash
# (a) the two stochastic v2.1 merges, pass 1 only (v21 is reused if still under /srv/stack_w)
VARIANTS="v21_attn_sr v21_full_sr" PASS2="" nohup bash /srv/stack/scripts/v3stack/stack_test.sh > /root/logs/stack_sr.log 2>&1 &
# (b) the stock pair, pass 1 only
VARIANTS="stock stock_full_sr" PASS2="" nohup bash /srv/stack/scripts/v3stack/stack_test.sh > /root/logs/stack_stock.log 2>&1 &
# or append the stock pair to the default list: STOCK=1 PASS2="" ...
```

Each new variant needs about 21 GB under `/srv/stack_w`: a 19.5 GB shard copy for a merge, or 20.2 GB for the stock
download. Before every fetch and merge, the script checks free space. If it is short, it prints which unused variant
directories to delete instead of failing halfway, for example `rm -rf /srv/stack_w/v21_full
/srv/stack_w/v21_full.merge_report.json` once v21_full is no longer needed. `MERGE_SEED` (default 0) sets the
rounding seed. The same seed gives byte-identical checkpoints.

A stochastic full merge does about twice the CPU work of a nearest one, because both modes are measured. Expect about
40-80 min on half the cores. Run the variants in the order shown, so the full merge overlaps with serving the
attention-only or stock variant. Pass 1 takes 0.75-1 h per variant, so (a) is about 1.6-2.2 h ($10-13) and (b) about
1.8-2.3 h including the stock download ($11-14).

## Time and cost (hard cap $40 at about $6 per GPU-hour, so about 6.5 GPU-hours)

| Stage | Wall time (est.) | Cost |
|---|---|---|
| setup: image pull, v2.1 download, selftest | 0.3 h | $2 |
| merges (CPU, overlap with v21 serving; attn ~5 min, full ~20-40 min) | 0 h extra | $0 |
| pass 1, per variant: server start ~5-10 min + agent_hard_noforce 25-40 min + ubench 220 ~10 min | 0.75-1 h | $5-6 |
| pass 1, three variants | 2.3-3 h | $14-18 |
| pass 2, per variant: restart + main ~8 + hardmath ~12 + code ~5 + rp ~12 min | 0.7 h | $4-5 |
| pass 2, three variants | 2-2.3 h | $12-14 |
| **total** | **4.6-5.6 h** | **$28-34** |

All estimates are UNVERIFIED: no suite has run on this engine yet. Stop the pod at 6 h ($36) whatever state the
run is in. Everything is resumable, so a second rental only redoes the unfinished suite.

If it runs long, drop in this order:
1. **`v21_full`**, when its merge report shows survival near 0 (`FULL_MIN_SURVIVAL=0.05` skips it automatically),
   or at least its pass 2 (`VARIANTS="v21 v21_attn"` for a pass-2-only rerun). Saves 0.7-1.7 h.
2. **rp** (`PASS2="main hardmath code"`). Saves about 0.6 h.
3. **code**, then **hardmath**.

Never drop `agent_hard_noforce`, `ubench` or the `v21` reference: together they answer the question.

## Fallback: runtime LoRA (full-precision expert delta)

Use this if the `attn_experts` merge fails its zero-delta byte check, or if survival shows the 4-bit experts lose the
delta and the full adapter must still be tested. It uses the custom vLLM path in scripts/v3rl (pod_setup.sh and
eval_ckpts.sh's `serve_one`), which has never run. Its own UNVERIFIED notes apply.

```bash
MODELS=v21 bash /root/inkling-scripts-src/v3rl/pod_setup.sh          # v2.1 -> /root/w/v21_merged, SkyRL vLLM env
# serve v2.1 + the adapter (eval_ckpts.sh serve_one, adapter kind), then run_suite.py --model stack --tag v21_lora
```

UNVERIFIED in that path: the custom vLLM accepts the training service's expert LoRA layout (shared lora_A [1,32,2304]
for w1/w3, shared lora_B [1,2304,32] for w2) without conversion. It also has to load and serve `--max-lora-rank 32`
on one GPU.

## v3 GGUF build (`gguf_v3.py`, `gguf_v3.sh`)

The v2 and v2.1 GGUFs were made by byte-patching 120 BF16 attention tensors into ggml-org's BF16 GGUF. v3 also changes
the experts, so `gguf_v3.py` patches those too, in four steps: `map`, `prove`, `patch`, `verify`. The quants then use
the same `llama-quantize` flags as rel21.sh. `--selftest` runs offline on a synthetic checkpoint and a synthetic GGUF.
It covers one NVFP4 tensor, one MXFP8 tensor and the attention tensors, and includes negative cases.

**Layout, from the official header.** The experts are BF16 tensors. `blk.L.ffn_gate_up_exps.weight` has numpy shape
`[128, 2304, 2304]` and `blk.L.ffn_down_exps.weight` has `[128, 2304, 1152]`. Both are `[expert, out, in]`, the same
order as the checkpoint's `w13 [E, 2I, D]` and `w2 [E, D, I]`.

- The down tensors are not square, so their shape fixes the orientation.
- The gate_up tensors are square, and the checkpoint interleaves the gate and up rows. `prove` therefore tries 3 row
  orders, with and without a transpose, 2 NVFP4 decode op orders and 2 BF16 roundings. It accepts only an exact byte
  match.
- Attention uses make_map_v11b.py's five rules plus `wr_du → attn_rel`.

The map has 192 pairs: 144 attention, 40 NVFP4 expert and 8 MXFP8 expert tensors. `--base-ckpt` is always STOCK,
even for a merge built on v2.1. That way v2.1's own attention edits get patched too. Expect 144 attention tensors for
a v3 adapter that covers all six modules.

Commands, on the pod. They need torch, safetensors and numpy, and do not use gguf-py.

```bash
# 0. local, offline
python3 scripts/v3stack/gguf_v3.py --selftest                                         # ends SELFTEST_OK
# 1. inputs (pod): official BF16 GGUF + imatrix, stock checkpoint, llama.cpp @19bc41dbb (a CPU build is enough)
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-BF16.gguf imatrix.gguf --local-dir /data/base
hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 \
    --include 'model*.safetensors*' '*.json' --local-dir /data/orig
git clone -q https://github.com/ggml-org/llama.cpp /data/llama.cpp && cd /data/llama.cpp && git checkout -q 19bc41dbb \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-quantize llama-server llama-perplexity
# 2. prove once (the runner also does this): sample first (~2 min), then full (~15-25 min, reads ~56 GB)
cd /srv/stack/scripts/v3stack
python3 gguf_v3.py prove --base-ckpt /data/orig --gguf /data/base/Inkling-2-Mini-BF16.gguf --out /data/v3gguf/proof/proof_sample.json
python3 gguf_v3.py prove --base-ckpt /data/orig --gguf /data/base/Inkling-2-Mini-BF16.gguf --out /data/v3gguf/proof/proof.json --full
#    must end PROVE_PASSED; "decision" in proof.json names the row order and rounding for each class
# 3. one ladder per merged checkpoint (patch, verify, 6 quants, SHA256SUMS/SIZES/MANIFEST.tsv); resumable
bash gguf_v3.sh /data/orig /srv/stack_w/stock_full_sr /data/v3gguf/stock   Inkling-2-Mini-v3
bash gguf_v3.sh /data/orig /srv/stack_w/v21_full_sr   /data/v3gguf/unbound Inkling-2-Mini-Unbound-v3
grep -E 'rc=|DONE|MISSING' /data/v3gguf/*/logs/gguf_v3.log
```

Under the hood, step 3 runs:

```bash
python3 gguf_v3.py patch --base-ckpt STOCK --merged-ckpt M --gguf OFF --proof proof.json --out $N-BF16.gguf --name $N --report patch_report.json
python3 gguf_v3.py verify --gguf $N-BF16.gguf --official OFF --merged-ckpt M --proof proof.json --out-dir verify   # VERIFY_PASSED
llama-quantize $N-BF16.gguf $N-Q8_0.gguf Q8_0
CUDA_VISIBLE_DEVICES="" llama-quantize --imatrix imatrix.gguf --tensor-type attn_q=q8_0 --tensor-type attn_k=q8_0 \
  --tensor-type attn_v=q8_0 --tensor-type attn_output=q8_0 --tensor-type attn_gate=q8_0 $N-BF16.gguf $N-$T.gguf $T 6
#   T in Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M, all five in parallel
sha256sum $N-*.gguf > SHA256SUMS; ls -l $N-*.gguf > SIZES      # MANIFEST.tsv rows: file, bytes, sha256
```

What to check in the outputs:
- `patch_report.json` has one row per replaced tensor.
  - Attention: `frac_changed_vs_official` should be close to 1.
  - Experts, after a stochastic-rounding merge: a substantial fraction of elements should change. The reference is
    the merge report's changed bytes / elements.
  - `base_mismatch` must be 0 everywhere, and `worst_rel_err` must be ≤ 2^-8 (BF16 round-to-nearest-even).
- `verify/verify.json` and `tensor_manifest.tsv` should show three things. Every mapped tensor equals the merged
  checkpoint pushed through the proven mapping. Every other tensor is byte-identical. Only `general.name` changed in
  the header.
- `patch` refuses to run without a passed proof for this exact GGUF header and mapping. `--force-unproven` overrides
  that with loud warnings. Never release a file built that way.
- rel21.sh's `KEEP` list does not include `attn_rel`. v2.1 never edited `wr_du`, but a v3 adapter does. Decide
  before the release whether to add `--tensor-type attn_rel=q8_0`. The tensor is small, 384×2304 per layer.

**Disk, RAM, time per variant.** Shared by both variants: the official BF16 (52.2 GB) and the stock checkpoint
(19.5 GB). Per variant: the merged checkpoint (19.5 GB), the patched BF16 (52.2 GB) and six quants (≈ 105 GB), so
about 180 GB. Keeping both variants takes ≈ 430 GB. If space is short, delete a variant's BF16 once its quants and
checksums exist.

`gguf_v3.py` holds at most about 2 GB in RAM (one 8-expert chunk in f32, in several copies) and streams everything
else. The five parallel quantisations need the same RAM as in rel21.sh, which ran them on a 93 GB pod.

Time per variant:
- patch: 15-25 min (a 52 GB copy, a 39 GB checkpoint diff, 48 expert tensors decoded twice)
- verify: 15-25 min (hashes 2 × 52 GB)
- quants: about 30-60 min on 30 cores

All of these estimates are UNVERIFIED, because nothing has run on a pod yet.
