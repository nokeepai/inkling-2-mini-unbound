# Inkling-2-Mini abliteration: progress report
2026-11-21, 12:05 UTC (T+2:34 since go at 09:31 UTC). Repo: ~/Code/inkling-abl

## Summary
- Setup, harness and diagnostics are done. The candidate search started at 11:54 UTC; the 8-GPU scoring fleet goes live at ~12:30.
- Spend: $10.51 of the $200 cap ($18.78 committed, burn $7.88/h). Forecast ~$100.
- ETA: first comparative results ~13:15, finalists ~14:45, final report ~17:05 UTC (inside 8 h).
- Key finding: OpenUncensored (OU) refuses only ~1% by phrase list, but ~21% once a judge catches its dodges and vague filler. Only ~50% of its answers are genuinely useful.
- Key lever: a per-channel conv after each attention output leaks 19–88% of a plainly removed refusal direction back into the model. OU's heavy edit (strength 2.5–4) works around this badly. We can remove it exactly.

## What was done, in order
1. Repo, brief (BRIEF.md), plan (PLAN.md), budget ledger (only ink-* pods; the account's nokeep production pods are untouched), append-only results file with locking.
2. Pinned everything: Inkling-2-Mini @ ed40e7da, OU @ 4e5f06d7, Transformers PR #51050 @ a10ebc3c, SGLang image sha256:421aed3a…
3. Data: 1,390 harmful prompts (AdvBench, HarmBench, StrongREJECT, SORRY-Bench base; mlabonne duplicates AdvBench), 6,000 harmless, 1,557 borderline (XSTest, OR-Bench-hard), 40 hand-written probes (honest limits, self-harm crisis). Deduped (exact + embedding cosine > 0.95), split 60/20/20 with seed 1234; the test split is sealed. SORRY-Bench was added after the owner accepted its gate; 8 non-harmful advice/opinion categories excluded.
4. Model recon: 24 layers, layers 6–17 looped 3× (48 visits). Loop controllers are shared across passes, so C5 is dropped. OU's edit decoded and reproduced bit-exactly (W − s·Q·QᵀW on 20 attention-output tensors, 2 directions). The residual stream is block-structured; attention output passes through a per-channel conv; original attention rows have unit norm (OU changes them up to 15%).
5. Literature scan: no published abliteration reaches ≤ 1% refusal at KL ≤ 0.02 (best ~0.024). Main recommendations: KL-constrained judge-in-the-loop search, norm-preserving ablation, thinking-position directions, protected-skill orthogonalisation.
6. Gate 0 (engine parity): failed as written, because SGLang agrees with itself only 95.2% top-1, below the 99% bar. No engine bug (loop passes clean). The owner approved recalibrated criteria; passed: HF vs SGLang top-1 95.6–96.0%, MMLU-300 77.0 vs 76.7, GSM8K-100 97 vs 99.
7. Fidelity harness (HF, KL): noise floor KL 0.0025 / top-1 98.2%, after finding that a 4-bit activation emulation inflated it to 0.016 (now off, per the brief). OU KL 0.064, top-1 91.6%. Scoring takes 12 s per candidate; a compact candidate format (~1 MB instead of 280 MB) is end to end in 29 s.
8. SGLang harness (refusal, capability): frozen engine on RTX 5090; verified weight swap and restore; 93-phrase refusal list built from the original's real declines. Original ×2 noise: refusal ≤ 0.4 pt, MMLU 1.7 pt. Transfer check passed (our C0 = OU).
9. Judge: StrongREJECT gemma-2b on GPU, ~1 min per candidate. Its agreement with my 50 hand labels is only 76–78%, so a stronger judge is being tested (result ~12:25). A batch-padding bug was found and fixed.
10. Diagnostics (Gate 2): refusal direction rank 2–3 from late visits. Harmfulness and refusal directions overlap (cos 0.43–0.54). Conv leak measured; an exact leak-proof edit exists (rank 4r per layer, 0 leak). Attribution (attention vs experts) rerun due ~12:20.

## Baselines (validation split)
| Metric | Original | OpenUncensored | Target |
|---|---|---|---|
| Refusal, phrase list (off / effort 0.9) | 97.9 / 98.2% | 0.4 / 1.1% | ≤ 1% |
| Refusal, phrase or judge | 98.6 / 98.6% | 23 / 21% | ≤ 1% |
| Useful answers (judge) | ~4% | 50 / 51% | ≥ 90% |
| KL vs original (HF) | 0.0025 (noise) | 0.064 | ≤ 0.02 |
| Top-1 agreement | 98.2% (noise) | 91.6% | ≥ 96% |
| MMLU-300 | 76.7–78.3 | 76.0 | within noise |
| GSM8K-100 | 99 | 98 | within noise |
| Over-refusal, borderline | 35.8% | 0.3% | ≤ original |
| Thinking-cap hits at effort 0.9 | 0.7% | 17.8% | — |
| Repetitive thinking at effort 0.9 | 1.4% | 29.5% | — |
| Honest-limit probes | 100% | 90% | at original's rate |
| Crisis-resource probes | 100% | 100% | at original's rate |

## Search (Phase 3), started 11:54 UTC
- Families: C1 = projected direction + leak-aware edit + norm preservation, with searched per-layer strengths; C2 = C1 + separation from harmfulness and protected skills (math, code, honesty, crisis); C3 = loop-aware ARA (starts ~12:20); C4 = expert adapter only if attribution shows > 20% expert share; C5 = dropped.
- First KLs, before tuning: 0.018–0.039 (vs OU 0.064).
- Leak-fix variants being compared: exact (rank 4r), svd2, svd1 (same size as the plain edit, leak 3–21% instead of 19–62%), plain.
- Scoring: KL worker (research pod), 8 SGLang workers (one 8×5090 node, all candidates on identical hardware), judge relay.

## Odds (current view)
- Beat OU on fidelity and answer quality: high (~85–90%).
- Beat IndrasMirror XQ's self-reported numbers (KL 0.028, 6% refusals): ~65%.
- Hit every brief target at once: ~30%; the hardest is ≥ 90% useful answers.

## Decisions made with the owner
- 10:28 UTC: Gate 0 recalibrated (Option A plus 6 additions: per-GPU-type pinning, frozen batch settings, noise floor as tie threshold, transfer check, top-1 diagnostic).
- SORRY-Bench gate accepted by the owner.

## Deviations (logged in PLAN.md)
- Update channel = this session. The orchestrator handles all pod lifecycle. No Anthropic key, so the open judge is used. sealed/ is protected by permissions and process only, with one OS user.

## Pods now
ink-research-1 (PRO 6000, diagnostics then ARA), ink-research-2 (PRO 6000, KL + judge), ink-eval-2 (5090, retiring), ink-fleet-1 (8×5090, booting). ink-eval-1 removed. All have TTLs ≤ 2 h.
