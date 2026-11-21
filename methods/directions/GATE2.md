# Gate 2: directions and diagnostics (agent `diag`, 2026-11-21)

All numbers: HF PR loader on ink-research-1 (W4A16, BF16), **train split only**. Visit index = **call order** v = 0..47 (MODEL_NOTES §1) unless stated.

## 1. Conv leak. Verdict: a conv-aware edit IS needed
`leak.json` (50 harmful train prompts, effort 0, all prompt tokens plus the forced prefix) and `leak_exact.json` (25 prompts).
- **Plain projection on wo_ud (s = 1), local leak:** ‖Qᵀy‖/‖Qᵀy_orig‖ at the post_attention_convolution output.
  - Range is **19–88%** of OU's Q surviving the conv, median about 40%.
  - It is worst at block-boundary and early layers (L6 pass 2/3 0.86–0.88; L4–8 0.54–0.77; L17 pass 2 0.71) and lowest at L9/L11/L15 (0.19–0.26).
  - The end-to-end measurement (weights actually edited, s = 1) matches the local counterfactual within 0.01.
  - The counterfactual reproduces the module output to a relative error of 0.17%.
- **OU's actual edit (s = 4/2.5):**
  - The pre-conv Q component is scaled by 1−s (−3 / −1.5) as designed.
  - After the conv, ‖Qᵀy‖/‖Qᵀy_orig‖ is **0.64–3.1** where the ideal is 3.0 / 1.5.
  - The deviation from the ideal (1−s)·Qᵀy is 0.47–2.7× ‖Qᵀy_orig‖.
  - So OU's reflection is far from a clean reflection once it passes the conv.
- **Hook on post_attention_convolution output:** 0.08–0.34% survives, which is BF16 rounding only.
- **Exact fix:** project S = span{Q, D₀Q, D₁Q, D₂Q, D₃Q} out of wo_ud's output space. D_k = diag(conv tap k), and the conv's identity term needs Q itself.
  - **Rank 10 = 5r for r = 2, on every layer.** Survival is **0.0000** at all 48 visits (fp32 local counterfactual).
  - The cost is that 10 directions per layer are removed instead of 2. The KL impact is untested.
  - The cheaper alternative is to bake the edit as a projection on the conv output, which is not a weight edit. Another option is a rank-r least-squares compromise. Method agents should treat rank 10 as the exact reference.

## 2. Directions (`directions.pt`)
- **Data:** 384 harmful (stratified by category, seed 1234) and 384 harmless (seeded) prompts at effort 0, plus 256+256 at effort 0.9 (`meta.efforts` = [0, 0.9]).
  - Effort 0.5 was skipped for time.
  - Thinking-token states are the mean of up to 16 greedy thinking tokens, teacher-forced after `<|message_model|><|content_thinking|>`.
- **Kinds:**
  - `out`: the decoder-layer output, which is a partial block sum.
  - `attn`: the post_attention_convolution write.
  - `mlp`: the post_mlp_convolution write.
  - `mlpin`: the mlp_input reader.
- **Positions:** last instruction token, `<|end_message|>`, `<|message_model|>`, `<|content_thinking|>`, and the thinking mean.
- **Keys:**
  - `means[e][kind]` holds `harmful` and `harmless`, each [48,5,2304].
  - `refusal_raw` is the raw DoM. `refusal_proj` is the DoM with the harmless mean projected out.
  - `refusal_post` and `refusal_think` are unit [48,2304]. `harmfulness` is unit, taken at the last instruction token.
  - `atoms[0.0][math|code|honest|crisis][kind]` is the set mean minus the harmless mean, [48,5,2304]. Math is 50 GSM8K questions, the others 30 written prompts each.
    - **Caveat:** atoms have no thinking position. Index 4 is zero, so use 0–3.
  - `Q` is the chosen basis [2304,3]. `Q_slot` and `bases` hold the alternative bases.

**Cosines (kind `out`, effort 0, mean over visits 24–47, with min–max):**
| | out | attn write | mlp write | mlpin |
|---|---|---|---|---|
| harmfulness vs refusal (post-instr) | 0.43 (0.18–0.64) | 0.46 | 0.33 | 0.57 |
| harmfulness vs refusal (thinking) | 0.54 (0.27–0.81) | 0.54 | 0.39 | 0.65 |
| post-instr vs thinking refusal | 0.51 (0.15–0.76) | 0.49 | 0.41 | 0.61 |

- **Harmfulness and refusal are far from orthogonal here** (0.4–0.55, versus SRA's ≈ 0.1), so C2's orthogonalisation against harmfulness will remove a large part of the direction. Make the ridge λ a search parameter, as planned.
- **Post-instruction and thinking refusal only partly agree** (about 0.5), so both are included in the basis (FINDINGS idea 5).
- **The lowest cosines sit at block-boundary visits** (v = 18, 30, 42; partial-sum restarts), as MODEL_NOTES predicted.

## 3. Rank and visits (`spectrum.json`)
**SVD of the 48 stacked unit refusal directions** (post + thinking) over call-order visits 24–47, kind `out`: energy 0.34 / 0.13 / 0.08 / 0.08 / 0.05. **The chosen rank is 3** (rule: add a component while its energy is ≥ 8%).
- **KV-slot reading of "24–47"** (= call-order v18–41, passes 2 and 3): energy 0.41 / 0.13 / 0.09 and also rank 3. Its principal cosines with the call-order basis are 0.999, 0.978, 0.26. **The first two directions are the same under either indexing**, so the ambiguity barely matters for r ≤ 2.
- **Overlap with OU's Q** (principal cosines): 0.76 and 0.57.
- **Share of each visit's refusal direction inside span(Q):** 0.69 (post) and 0.78 (thinking) on v24–47, but only 0.49 on v0–23. Q is a late-visit subspace.
- **Coverage per category** (share of the category's projected DoM, positions 1–4 over v24–47, that lies in span(Q); and its projection relative to the all-harmful mean):

| category | n | share in span | proj vs all |
|---|---|---|---|
| crime_illegal_goods | 128 | 0.60 | 1.05 |
| violence | 32 | 0.59 | 1.01 |
| sexual | 27 | 0.58 | 1.00 |
| hate_harassment | 47 | 0.57 | 0.96 |
| other_harm | 16 | 0.57 | 1.00 |
| misinfo_deception | 58 | 0.56 | 0.91 |
| cyber | 57 | 0.55 | 1.01 |
| cbrn_weapons | 19 | 0.53 | 1.03 |

Coverage is even across categories. The weakest are misinfo (projection 0.91 of the mean) and CBRN (0.53 share); watch these in validation.

## 4. Attribution (`attribution.json`, `attribution_gens.json`)
- **Setup:** rerun at 11:50Z with 60 harmful train prompts at effort 0. Thinking was capped at 256 tokens, with 96 answer tokens, greedy decoding. Q is rank 3 (directions.pt).
  - Q was projected with hooks out of the writer's conv output, at all 24 layers and all positions.
  - Refusal is scored by the seed phrase list on the final answer.
  - This replaced the first run (100 prompts × 7 conditions), which ran out of memory.

| condition | refusal | Δ vs base | no answer by the cap | thinking past the cap | mean thinking length |
|---|---|---|---|---|---|
| base | 0.983 | n/a | 0 | 1 | 96 |
| attention writes only | 0.233 | 0.750 | 8 | 15 | 200 |
| MoE writes only | 0.067 | 0.917 | 6 | 11 | 203 |
| both | 0.000 | 0.983 | 7 | 20 | 219 |

- **Expert-path share of the joint effect:**
  - Δ_MoE / Δ_both = **0.93**.
  - Δ_attn / Δ_both = 0.76.
  - The effects overlap and are not additive. Normalised, the MoE share is Δ_MoE / (Δ_attn + Δ_MoE) = **0.55**.
  - **Every definition is above 20%, so C4's gate is met.**
- **Caveats:**
  - Keyword scoring on n=60, with no judge.
  - The ablated answers often comply evasively or quietly rewrite the request (a joke instead of an insult, a generic access-control snippet). The judge will count some of these as refusals or non-specific answers.
  - Ablation more than doubles thinking length, and 11–20 of 60 rows run past the 256-token cap, so the no-answer rows are not scored as refusals.
  - Projecting out the MoE write is a hook. It is not a weight edit, because the experts are NVFP4. C4 has to reach that path through a BF16 adapter or by reading the router/MoE input, not by editing experts.
  - The layer-group split was not run (time).

## 5. Loop controllers
**Confirmed: they are not per pass.** Layers 6–17 each have one set of q/k/v/mlp message gates (rank 64), shared by passes 2 and 3. Pass 1 has no messages (MODEL_NOTES §5, M:1397-1419). C5 has no pass-specific weights to edit and should not run.

## 6. Recommendation for method agents
- **Writer:** use the attention write (wo_ud → post_attention_convolution).
  - A plain wo_ud projection leaves 19-88% of Q in the residual, so a weight edit must use the conv-aware subspace S = span{Q, D₀..D₃Q}: rank 5r, exact.
  - Alternatively, search a reduced-rank compromise against KL.
  - For hook-based search trials, project on the post_attention_convolution output. That is exact, and it is the reference the baked file must match.
- **Direction:** use `Q` (rank 3) from directions.pt: post-instruction plus first-thinking refusal DoM, harmless-mean projected, over call-order v24-47.
  - Start the search at rank 2. The first 2 directions are robust to the visit-index ambiguity; the 3rd is not (cos 0.26 between the call-order and KV-slot readings).
  - For C2, ridge it off `harmfulness` and the atoms. Harmfulness overlap is high (cos 0.4-0.55), so λ must be searched; don't fully orthogonalise.
- **Layers:** L4-23 kernel as planned, with a loop multiplier. Q is a late-visit subspace (it captures 0.69-0.78 of v24-47's refusal directions vs 0.49 of v0-23).
  - Avoid L18-23 by default only if KL binds, because those are visits 42-47, where capture is still high (0.64-0.90).
- **Positions:** all tokens (standard).
  - Directions use the template and first-thinking positions; those give the cleaner refusal signal.
  - Thinking-position directions are the most consistent across efforts (capture 0.78 at e0 vs 0.77 at e0.9; post-instruction 0.69 vs 0.64).
- **C4 is justified.** The MoE write carries as much refusal signal as attention, or more. An attention-only edit (C1–C3) removed it from only 75% of prompts here (keyword). Expect attention-only families to stall in the low refusal tens unless the strength is raised, which costs KL. Give C4 a low-rank BF16 adapter on the MoE output path, or a projection on post_mlp_convolution baked as a dense BF16 op if the format allows. Never touch NVFP4 tensors.
