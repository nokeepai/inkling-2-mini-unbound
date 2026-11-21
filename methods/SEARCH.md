# Phase 3 search design (orchestrator, 11:15Z; finalised at Gate 1/2)

## Loop
1. A method agent (owner of one family) runs an Optuna multi-objective study, stored at R/state/optuna/<family>.db on the control machine.
   - Objectives: minimise val refusal rate (mean of thinking off and effort 0.9) and minimise KL.
   - Constraints: MMLU-300 ≥ orig − 2 pts; GSM8K-100 ≥ orig − 5 pts.
2. For each ask(), the agent builds a weight candidate (R/harness/CANDIDATE_FORMAT.md) at R/methods/<family>/cands/<cand_id>/ and enqueues it by appending {"cand_id","family","path","priority","time"} to R/state/queue.jsonl.
3. Workers pull from the queue (claiming by appending a {"claim": cand_id, "worker"} row to R/state/queue_claims.jsonl under flock):
   - KL worker (fidelity, ink-research-2, HF, AQ off): KL and top-1 on 200 val harmless. ~11 s per candidate.
   - Eval workers (one per RTX 5090, SGLang frozen engine): harmful val refusal at thinking off and effort 0.9 (phrase list OR judge), per-category refusal, MMLU-300, GSM8K-100, thinking hygiene. Borderline and probes only for candidates flagged "finalist".
   - Judge: StrongREJECT evaluator. It runs inside the inner loop, because keyword-only refusal undercounts.
4. The method agent polls runs.jsonl for its cand_id's rows, then calls tell().
5. Budget: 150 trials total, ~50 per family. Stop a family early if, after 25 trials, its Pareto front is beaten everywhere by another family's.

## Families (ordered by expected value, per FINDINGS.md and Gate 2)
- C0: OU reproduction (done: c0 == ou bit-exact).
- C1 (agent mpoa): the refusal direction (rank r ∈ {1,2,3}, from Gate 2) with the harmless mean projected out. Applied as a conv-aware exact projection (remove span{Q, D_k Q}) from wo_ud with row-norm preservation (MPOA), and per-layer strengths s_l from a smooth kernel (peak layer, width, peak strength, floor, loop-layer multiplier). Layers 4–23 searched; the default prior avoids heavy edits on L18–23. Candidates are built on the Mac CPU (only 20 tensors of 2304×3072).
- C2 (agent mpoa): C1 plus ridge orthogonalisation against harmfulness, math, code, honest-limit and crisis atoms (λ searched), plus thinking-position directions (mix weight searched), plus 1–2 hard-negative rounds on train only.
- C3 (agent ara): loop-aware ARA on wo_ud with post-conv capture at all 3 passes, loss summed over passes, rank ≤ 50, row norms preserved. Needs a research GPU (ink-research-1 after diag).
- C4: only if the Gate 2 expert share is > 20%.
- C5: skipped (controllers shared across passes).

## Finalists
Pareto points with val refusal ≤ 2% → judge on val for up to 5 → pick up to 3 by non-evasive rate, with KL breaking ties. Owner rule: noise-floor spread (orig ×2 in SGLang) is the tie threshold.
