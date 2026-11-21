# Fidelity harness config (HF, frozen once sealed)

- **Engine:** Transformers PR #51050 at a10ebc3c (pins.json).
  - Experts are dequantised to BF16 at load, with BF16 activations (**W4A16**).
  - **act_quant = False.** This is a hard default. kl_eval.py exits if AQ=1 unless `--supplementary` is passed.
  - cands.load_candidate refuses a model with expert activation fake-quant patched in (unless FID_SUPPLEMENTARY=1).
  - Act-quant ON numbers are supplementary only.
- **Expert forward:** the PR's own expert loop. The fast W4A16 path failed its exactness test (max logit diff 19.6) and was removed from the scripts.
- **GPU and attention:** RTX PRO 6000 (ink-research-2). Eager attention (the PR disables SDPA). PyTorch reference causal conv.
- **Canonical rows:** `engine: hf_w4a16`.
  - Rows with `act_quant: true` (10:51–11:02Z) are superseded.

## Reference set (regenerated with act_quant=False, 11:05–11:13Z)
- **KL prompts:** 200 harmless val prompts, `random.Random(1234).sample(range(1200), 200)` over data/val/harmless.jsonl, in file order. The ids are in kl_val_ids.json.
- **Continuations:** the original's greedy output at effort 0, at most 256 tokens, thinking included, truncated after eos.
  - Generated in left-padded batches of 100, prompts sorted by length.
  - 40,280 positions; 113/200 end in eos.
  - The original's teacher-forced argmax matches its greedy tokens 98.18% of the time.
- **Log-prob cache:** the original teacher-forced with right padding, batch size 8, prompts sorted by continuation length (descending).
  - Stored as a float16 log_softmax over the full vocabulary (V=200058), plus the fp32 argmax.

| File | sha256 |
|---|---|
| orig_cont_val.pt | bca7f587b182c666c7e1cd7024eebe9f95abd6f11b3842f383d320a95a508fca |
| kl_val_ids.json | b894302d06e9e9ebe44c4094943914328664148ecccaa41903a288518445fb4d |
| /srv/fid/orig_lp.f16 (pod only, 16.1 GB) | 894ba2e64a44985369adbe678a5ef914a8d3ee9fbcc2b7b0c0530fd03190684d |
| orig_top1.npy (pod) | 9e0ddac2db9d8131e2133fdee0beacfd42532a8c80af46a30f3a7806640c2cba |
| orig_cache_meta.json | 04d67b763578c6d9df420c399f414239adcb37523554417a9bba8fd15d76a8f4 |

## Scoring
- **Batching:** candidates are scored with exactly the cache's batching (bs 8, sorted, right padding, same order).
  - Orig against its own cache gives KL 3.5e-8 (fp16 cache rounding) and top-1 100%.
  - The worker checks this at startup.
- **KL:** KL(orig ‖ cand) per position, in fp32 from logits, token-pooled over all positions.
  - 95% CI from a bootstrap over prompts (10k resamples, seed 1234).
- **top-1:** the candidate's argmax equals the original's argmax.
- **Noise floor protocol:** KL(orig ‖ orig) with a batch-shape perturbation: orig at batch size 1 against the cache's bs-8 batching (`noise_bs1`).
  - Result: KL 0.00251, top-1 98.23%.
  - Supplementary: reshuffled at bs 8 gives 0.00259 and 98.20%.
  - KL differences below about 0.0025 count as ties.
- **Capability (HF):** see harness/cap/FORMAT.md.

## Worker (Phase 3, methods/SEARCH.md §Loop)
- **Pod server:** harness/kl_server.py on ink-research-2 (log /root/logs/kl_server.log).
  - It holds the model and the cache, and self-checks KL ≈ 0 at startup.
  - File RPC: /srv/cands/<cand_id>/ plus a READY nonce in, a row in /root/fid/kl_server_results.jsonl out.
  - Per candidate: swap in (bit-verified), score with the cache's exact batching, then restore (bit-verified).
  - A probe forward must then reproduce the startup logits exactly. If it does not, the server writes the row and exits.
- **Control loop:** harness/kl_worker.py on the Mac (log harness/fidelity/kl_worker.log).
  - It polls state/queue.jsonl every 15 s and claims candidates in state/queue_claims.jsonl under flock (worker `kl_worker@ink-research-2`, kind `kl`).
  - It rsyncs the folder, verifies the sha256 on the pod, and waits for the result.
  - It appends `kl` and `top1` rows (agent kl_worker, split val, engine hf_w4a16, CI, n, sha256, restore_probe_ok, pos_stats).
  - Server errors are written as `kl_error` rows. Transport errors retry 3 times first.
- **Compact candidates** (lowrank.safetensors) are built on the pod by harness/build_cand.py.
  - The worker syncs build_cand.py to /root/fid automatically.
  - Only spec.json and lowrank.safetensors are pushed.
  - The research-2 build writes the canonical sha256.txt, and the worker copies it back into the local candidate folder.
  - A different local sha256.txt is kept as sha256.local.txt and logged.
- **c0_lr test** (c0 in compact form: A = −s·Q, B = W32ᵀQ, no r): 864 KB. The research-2 build sha is 0bffeecd… and is reproducible.
  - It differs from OU in 345 of 141.6M elements (1 BF16 ULP, max 2.4e-4). Layers 4–5 and 18–23 (s=4) are exact; layers 6–17 (s=2.5) have 21–34 differences each, from operation order.
  - KL 0.06342 vs 0.06365, a gap well below the 0.0025 floor. 29 s end to end.
- **End-to-end test (11:45Z):** ou and c0 were both enqueued and both scored KL 0.06365, top-1 91.64%, with restore_probe_ok.
  - Each took 110–173 s end to end, almost all of it the 280 MB upload; scoring is about 12 s.
- **Superseded:** scripts/fidelity/kl_sync.py (an earlier version, now stopped).

## Pod replacement (12:28Z): ink-research-2 expired, now ink-research-2b (RTX PRO 6000 Blackwell Server Edition)
- Rebuilt from the LOCAL orig_cont_val.pt (bca7f587…), using the same batching. The new cache is **byte-identical**: orig_lp.f16 894ba2e6… and orig_top1.npy 9e0ddac2…. hf_name_map.json is identical, and ou/c0 are 3ce3c497….
- Re-measured on 2b: noise_bs1 KL 0.00251, top-1 98.23%; ou/c0 KL 0.06365, top-1 91.64%. Both are identical to research-2 to every printed digit. No candidate needs re-scoring.
- The current cache sha is in harness/fidelity/cache_sha.txt. Worker rows carry `cache`; rows without the field are on 894ba2e6….
- kl_worker: `KL_POD=ink-research-2b`. Claims by `kl_worker@ink-research-2` count as its own. Transport failures never write rows any more (they back off and retry).
  - The 4 `kl_error` rows from the outage (c3t001, c3t003, c3t004, c3t005; error "after 3 tries", no live pod) are ignored and those candidates are re-scored.
