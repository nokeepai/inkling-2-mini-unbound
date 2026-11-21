# Candidate format (shared by method agents, eval workers, fidelity and the auditor)

A candidate is a folder `R/methods/<family>/cands/<cand_id>/` (copied to pods as `/srv/cands/<cand_id>/`):

- `spec.json`: {"cand_id", "family", "parent" (null or cand_id), "created", "agent", "params": {...free-form search params...}, "kind": "weights"}
- `tensors.safetensors`: replacement tensors, keyed by the EXACT checkpoint tensor name from model.safetensors.index.json, BF16, same shape as the original. Only BF16 tensors may appear (wo_ud, post-attention conv weights, norms, router/controller BF16 weights). Never an NVFP4/MXFP8 expert tensor. An optional low-rank BF16 expert adapter (C4) gets its own format later, if C4 runs.
- `sha256.txt`: sha256 of tensors.safetensors.

### Compact form (preferred for search trials; 280 MB uploads were the bottleneck)
Instead of `tensors.safetensors`, a candidate may ship `lowrank.safetensors`. For each edited tensor name N it holds `N.A` [out,k] fp32, `N.B` [in,k] fp32 and optionally `N.r` [out] fp32 (row scale). The replacement is
  W' = bf16( diag(r) · (W_fp32 + A·Bᵀ) )   (r = 1 if absent), computed on CPU in fp32 by R/harness/build_cand.py.
`build_cand.py <cand_dir> <orig_weights_dir> <out_dir>` writes `tensors.safetensors` plus its sha256. Each pod builds locally from its own /srv/w/orig. The first build of a candidate (on ink-research-2) records the canonical sha in `sha256.txt`; other pods must reproduce it. On a mismatch, the worker fetches the canonical tensors pod-to-pod with `lium cp` and logs the mismatch.
Method agents: build W' yourselves however you like, then FACTOR the delta (W'/r − W) into A·Bᵀ exactly (it is low-rank for projection-type edits), choosing r for the row norms. Verify locally that build_cand reproduces your intended BF16 tensor bit-exactly before enqueueing.

Every candidate must be expressible as replaced BF16 tensors (that is what gets baked). Hook-only experiments are allowed during diagnostics, but trials in the search are weight candidates.

Baseline ids: `orig` (no tensors), `ou` (OU's 20 wo_ud tensors extracted from its checkpoint), `c0` (orig + edit.pt applied by our code; must equal `ou` bit-exact).

Loading:
- HF (fidelity, KL): load orig once, then for each candidate copy its tensors into the matching parameters in place (keep a CPU copy of the originals to restore). The name map from checkpoint name → HF parameter is in R/harness/hf_name_map.json (written by the fidelity agent).
- SGLang (eval workers): start the server on /srv/w/orig once, then for each candidate push the tensors with the server's weight-update endpoint (update_weights_from_tensor or update_weights_from_disk with a merged dir); restore the original tensors after. The worker must verify, by a fixed probe prompt's logits, that the swap took effect and that the restore returns to the original's logits exactly.
