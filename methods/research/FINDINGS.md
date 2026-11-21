# Scout findings (2026-11-21 ~10:05Z)
Saved by the orchestrator from the scout's message (its harness blocked the file write). Numbers come from search highlights plus a few full page reads; verify before relying on them. "(ours)" = the scout's own derivation.

## Frontier check
No source reports judged refusals ≤ 1% AND KL ≤ 0.02. Nearest:
- abliterix Qwen3.6-35B-A3B, attention-only: 1/100 keyword at KL 0.0242 (huggingface.co/eggdog100/Qwable-v1-Qwen3.6-35B-A3B-abliterated)
- Inkling XQ: 6/100 at KL 0.028
- SRA: first-token KL 0.025 mean, 0–2% refusals (arxiv.org/abs/2601.08489)
- SOM, 4–5 directions: 3/100 at KL 0.079–0.117 (github.com/p-e-w/heretic/pull/196)
- LoRA-ARA attention-only, Qwen3-30B-A3B: 92–99/100 at KL ≤ 0.033, 49 at 0.077, 1 at 0.346 (heretic issue 342)

KL 0.02 is the binding constraint. Plan a search, not a fixed recipe.

## Top 8 (rank = gain × feasibility)
1. **Judge-in-the-loop, KL-constrained selection** of visits, layers and strengths (C2 search). Pick argmin KL subject to judged refusal ≤ x, using a StrongREJECT-style specificity and coherence judge, not keywords.
   - KL varied about 80× among candidates with ~0 refusals.
   - A refusal-only pick cost −6.0 pp MMLU; a KL-aware pick gained +0.1 pp (github.com/dallen2021/refusal-direction-repro).
   - Keyword 3–5/100 became 33+/100 under an LLM judge (heretic PR 211).
   - Arditi filter: KL < 0.1, skip the last 20% of layers. OU edits L18–23 at s=4, inside that zone.
   - Search 6–8 kernel parameters over L4–23 plus a loop-layer multiplier.
2. **Post-conv-aware exact projection (ours).** The write is y_t = (h_t + Σ_k D_k h_{t−k})/√5, with per-channel diagonal D_k. QᵀW' = 0 does not give Qᵀy = 0.
   - Exact fix: remove S = span{D_k Q, k = 0..3} (rank up to 4r) from W's output space.
   - Step 0: measure ‖Qᵀy‖/‖y‖ after a plain edit; skip the fix if the leak is small. Risk: rank 8 costs KL.
3. **C1 as published:** projected direction plus row-norm-preserving ablation.
   - r_proj = r − (r·μ̂_hl)μ̂_hl; winsorise activations at 0.995; do the maths in fp32.
   - Gemma-3-12B NatInt: 21.33 vs 18.72 baseline vs 18.64 plain (grimjim blog). Heretic PR 52: magnitude preservation is why biprojection beats projection.
   - Caveat: the gain is in capability, not KL, and some models respond poorly. Re-tune s after renormalising.
4. **C2 protected-atom orthogonalisation,** including crisis-reply and "I don't know" atoms.
   - Ridge-regress Q off the harmfulness direction, the math/code/reasoning atoms, style confounds, crisis replies and honest-limit replies. Put gray/XSTest prompts on the harmless side.
   - SRA, Qwen3-VL-4B: KL 2.088 → 0.044 at the same 0% refusal. Harmfulness and refusal directions have cos ≈ 0.1 (arxiv.org/abs/2507.11878).
   - Make the ridge λ a search parameter.
5. **Thinking-aware capture.** Take DoM at post-instruction tokens AND at the first 8–16 CoT tokens (teacher-forced refuse vs comply traces). Include both if their cosine is low.
   - CoT-direction ablation left 0.4–7.5% refused, vs 4–10% at end-of-prompt (arxiv.org/html/2507.03167), but cost 5–30% relative on reasoning benchmarks.
   - Evaluate with the thinking regenerated, not fixed.
6. **Per-writer module-output directions, then ARA-lite on wo_ud (C3).** Use each wo_ud's own post-conv output per visit.
   - Full ARA: LoRA rank ≤ 50, L-BFGS, preserve + kNN-steer + overcorrect terms, row-magnitude preservation (heretic ara.py); sum the loss over 3 passes; the capture must include the conv.
   - Fallback if 1–5 stall at 3–6% refusals.
7. **Hard-negative refinement for soft and evasive refusals.** Generate, judge, add still-refusing or vague (specific < 3) prompts to the harmful side, recompute. Repeat for 2–3 rounds. Count "I can't, but here's a safe alternative" as refusal. Train and val only.
8. **2–3 directions under a KL budget.** SOM beat a single direction on 8/8 models (arxiv.org/abs/2511.08379), but multi-direction runs sit at KL 0.08–0.12. Drop this first if KL binds.

Speculative: read-side projection of Q out of the router gate and controller inputs.

## Specifics
- **Evasive answers.** The cause is a removed harmfulness belief (all-visit directions). Fixes: take the refusal direction at post-instruction/first-CoT positions, orthogonalised against harmfulness; use late visits; search strength with a specificity judge; add hard negatives. Never count garbled text as compliance.
- **Thinking.** Refusal lives jointly in activations and the thinking trace. Working recipes are attention-output edits plus a judge, e.g. R1-0528-Qwen3-8B Heretic: 4/100 refusals at KL 0.0413.
- **Looped models.** Later loops re-derive content and undo one-loop patches (arxiv.org/abs/2609.01924). A shared-weight edit acts on every pass, which is good. Inkling restarts each pass from the pre-loop state, so pass-1 effects reach later passes only through messages and KV. If refusals persist, check whether Q survives in the controller outputs.
- **Block residual.** No literature. Take DoM at mlp_input/attention_input outputs, or per-writer outputs. Boundary visits (0, 6, 12, 18, 30, 42) are not comparable.
- **MoE.** Routing carries some safety signal (SteerMoE; arxiv 2602.08621, 2610.02910), but attention-only editing suffices for abliteration.
  - flay ran 8 expert/router experiments with no gain; masking "refusal experts" raised refusals to 95% (github.com/nooesc/flay).
  - C4 adapter = late option.

## What not to do
1. Use a keyword refusal metric or first-token KL alone (eggdog100 v1: KL 0.0144, yet the model loop-collapsed).
2. Select directions on refusal alone (−6 pp MMLU).
3. Make aggressive router or expert edits.
4. Use all-visit directions.
5. Assume a single direction is free on a thinking model.
6. Add many directions without a KL budget.
7. Assume norm preservation helps everywhere.
8. Assume results transfer across releases.

## First change
Run the 10-minute leak measurement on the OU edit (idea 2, step 0). Switch selection to a judge-in-the-loop, KL-constrained search (idea 1) on top of C1 (idea 3).
