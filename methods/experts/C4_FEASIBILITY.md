# C4 feasibility: the MoE / expert path (agent `experts`, 2026-11-21)

HF PR loader (a10ebc3c), act_quant OFF, BF16. Train split for localisation and refusal; 50 harmless val prompts for KL.
Pod: ink-research-1 (ink-research-3's HF download stalled at 134 MB). Scripts: R/scripts/experts/.

## Q1. Which existing MoE-path tensors are editable, and what can they express?

MoE write at layer i, token t (M:1052-1062, 1152-1172, 775-811, 329-393):

    x_t   = post_attention_layernorm( mlp_message_gate( mlp_input(blocks, partial) ) )      # BF16 / F32 params
    s_j   = sigmoid(W_r[j] · x_t);  sel = top8(s + b);  g_j = 8 · s_j / Σ_sel s                # W_r BF16, b F32
    m_t   = Σ_{j∈sel} g_j · E_j(x_t)                                                        # E_j NVFP4/MXFP8: frozen
    y_t   = ( (1+w3)⊙m_t + w2⊙m_{t-1} + w1⊙m_{t-2} + w0⊙m_{t-3} ) / √5                      # mlp_sconv taps, BF16

What we need for the hook result (MoE-only exact projection, refusal 98% → 7%): y_t ← (I − QQᵀ) y_t, a dense
rank-3 operator on the write. Every editable tensor on this path, and the strongest thing it can express:

| tensor (ckpt) | dtype | where | can express | projection of Q from the write? |
|---|---|---|---|---|
| `layers.i.mlp_sconv.weight` [2304,1,4] | BF16 | after experts | per-channel, per-lag scale: y = Σ_j diag(c_j) m_{t-j} | **No.** Diagonal. Best diagonal approximation of I−QQᵀ is diag(1−‖Q_c‖²); it leaves ≈ all of the Q energy (see diag_math.json). Zeroing whole channels removes Q only in proportion to their share of Q, which is spread over all 2304 dims |
| `layers.i.mlp.gate.weight` [128,2304] | BF16 (fp32 in model) | router logits | which experts fire and their mixing weights, as a function of x | **No**, but can drop or down-weight specific experts, conditionally on x (logit_j −= α·hᵀx). Removing expert j hands its slot to the 9th expert, and g renormalises |
| `layers.i.mlp.gate.bias` [128] | F32 | selection only | per-expert, input-independent exclusion/inclusion (bias = −1e4 removes expert j everywhere, all passes) | **No**; unconditional routing change |
| `layers.i.mlp_norm.weight` [2304] | BF16 | MoE input | per-channel input scale (router and experts both read it) | **No**; diagonal and on the input side; experts are nonlinear |
| `attn_res.mlp_queries.N` | F32 | mlp_input block attention | which blocks the MoE reads (softmax over ≤ 5 RMS-normed blocks) | **No**; a mixing choice over a handful of states |
| `controllers.mlp.{down,forget,gain}` (L6–17) | BF16 | MoE input, passes 2–3 only | (1−f)⊙h + g⊙msg, f,g = tanh(low-rank(h,msg)) | **No**; elementwise gates on the input; pass 1 untouched |

Conclusion of the maths: **no existing tensor can remove span(Q) from the MoE write.** The only dense matrices
on the path are inside the experts (NVFP4/MXFP8, frozen). Everything else is diagonal, a softmax mixing choice, or
routing. In-format options are therefore limited to (a) routing edits on the experts that carry refusal and (b)
attacking what the MoE *reads* via the attention edit (already C1–C3 territory). The exact projection needs a new
operator: either a rank-3 op after `mlp_sconv` (y − QQᵀy) or, equivalently before the conv, a rank-12 shared LoRA on
every expert's `down_proj` (ΔW2_e = −U Uᵀ W2_e, U = basis of span{c_j⊙Q}, the conv-aware subspace; the delta is a
BF16 adapter beside the NVFP4 weight, never a re-quantisation).

**Diagonal edits, measured on Q (directions.pt, rank 3; `out/diag_math.json`):** the largest single channel holds
0.86% of Q's energy. The best diagonal approximation of I−QQᵀ leaves **99.4%** of Q's energy. Zeroing the top 100
channels' current-token tap removes only 21%; 50% needs about 450 channels and 77% needs 1000, which would wreck the
write. The conv taps and `mlp_norm` are ruled out as a projection.

## Q2. Expert localisation (`localise.py`, `analyse_loc.py`, `ranking_small.json`)
- **Setup:** 192 harmful and 192 harmless train prompts (seed 1234), effort 0, with the forced prefix. Each routed
  expert's weighted output is pushed through the MoE conv exactly (it is linear per channel) and projected on Q.
  Score = DoM (harmful − harmless) of each expert's Q-write, summed over the 3 post-instruction positions, signed
  along that visit's total MoE refusal write. Check: the per-expert sum reproduces the hooked conv output to 1.3e-4
  (relative).
- **By layer, the write is concentrated in the loop core, layers 9–11:** L11 27.2%, L10 22.7%, L9 13.7% (64% together);
  L8 5%, L13/15 4% each, everything else ≤ 3%. Layers 0–5 and 20–23 carry almost nothing (< 3% in total).
- **By expert (physical layer, expert; all 3 passes summed):** top 8 pairs = **41%**, top 16 = 53%, top 32 = 66%,
  top 64 = 81%, top 128 = 90%. Per visit, the top 8 experts carry 91% of that visit's write (on average).
  Per (visit, expert): top 8 = 25%, top 32 = 47%, top 128 = 76%.
- **Top pairs, with selection rate harmful / harmless at the post-instruction positions:**

| layer.expert | share | sel harmful | sel harmless |
|---|---|---|---|
| L10.e32 | 12.8% | 0.88 | 0.17 |
| L9.e39 | 7.7% | 0.87 | 0.70 |
| L11.e53 | 6.0% | 0.94 | 0.24 |
| L11.e18 | 3.2% | 0.80 | 0.77 |
| L10.e19 | 2.9% | 0.51 | 0.40 |
| L11.e14 | 2.9% | 0.79 | 0.72 |
| L11.e22 | 2.8% | 0.58 | 0.58 |
| L9.e94 | 2.3% | 0.83 | 0.34 |

- **Reading:** two mechanisms. A few experts are **routed** mainly on harmful inputs (L10.e32, L11.e53, L9.e94:
  selected 3–5× more on harmful), which a router edit can reach. Others are selected about equally on both and
  **write refusal only because their input differs** (L9.e39, L11.e18, L11.e22). A router edit cannot remove these
  without also removing them on harmless text.

## Q3/Q4. Hook prototypes: refusal and KL (`hooks_rows.py`, `pod/hooks.jsonl`)
- **Setup:**
  - **Refusal:** 60 harmful train prompts (diag's set: Random(7) shuffle, first 60), effort 0, thinking cap 256,
    96 answer tokens, greedy. Scored with the seed phrase list (as diag) and the full phrase list
    (harness/refusal_phrases.txt), on the first 300 characters of the final answer. Rows without an answer by the
    cap count as not refused, as in diag; `no_ans` is reported.
  - **KL:** 50 harmless val prompts (Random(1234)), orig greedy continuation, **≤ 160 tokens** (shortened from 256
    for time), teacher forced, token-pooled KL(orig‖cand) and top-1. Base vs itself: 3e-8.
  - Conditions are row-masked hooks (each batch row carries its own condition), HF PR loader, AQ off.
  - All projections are s = 1 with diag's rank-3 Q, on every token. **These are not tuned candidates.** Compare
    them with each other, not with the 0.02 target.
- **Conditions:**
  - `attn`: exact Q projection on the post_attention_convolution output, all layers. This stands in for an exact4
    attention edit (C1).
  - `moepost`: exact Q projection on the post_mlp_convolution output, all layers. This is the rank-3 adapter and the
    upper bound.
  - `moepostL`: the same, on layers 8–11 only.
  - `mask32`: router bias −1e4 on the top-32 pairs. This is in-format (F32 `gate.bias`).
  - `cond32_d6`: router weight rows of the top-32 pairs minus α·h, where h is the harmful-vs-harmless DoM of the
    MoE input, made orthogonal to the harmless mean. α is set so that the harmful-mean logit drops by 6. This is
    in-format (BF16 `gate.weight`).
  - `topexp8`: S removed from only the top-8 pairs' outputs. This is an adapter on 8 experts.

- `rprojK` (the lead's idea): project span{h_l, Q} out of the router rows of the top-K pairs only. This is in-format
  (BF16 `gate.weight`, rank ≤ 4 per layer). Q is used in residual coordinates, which approximates the router-input
  coordinates (mlp_norm rescales the channels).

| condition | in format? | refusal (seed / full list) | no answer | KL | top-1 |
|---|---|---|---|---|---|
| base | n/a | 93% / 95% | 1 | 0.000 | 1.000 |
| attn | yes (C1) | 22% / 23% | 4 | 0.050 | 0.920 |
| moepost (rank-3 adapter, all layers) | **adapter** | 10% / 8% | 5 | 0.109 | 0.885 |
| attn+moepost | adapter | 0% / 0% | 7 | 0.196 | 0.844 |
| moepostL (adapter, L8–11) | adapter | 58% / 58% | 5 | 0.035 | 0.941 |
| attn+moepostL | adapter | 0% / 0% | 7 | 0.095 | 0.894 |
| attn+topexp8 (adapter on 8 experts) | adapter | 7% / 3% | 7 | 0.058 | 0.913 |
| mask32 (bias) | yes (F32) | 53% / 42% | 17 | 0.215 | 0.837 |
| attn+mask32 | yes | 3% / 2% | **31** | 0.239 | 0.824 |
| attn+cond32_d6 | yes (BF16) | 5% / 5% | 6 | 0.072 | 0.906 |
| attn+rproj8 | yes (BF16) | 17% / 13% | 6 | 0.052 | 0.920 |
| attn+rproj32 | yes (BF16) | 20% / 18% | 4 | 0.052 | 0.920 |
| rproj32 | yes (BF16) | 92% / 92% | 1 | 0.005 | 0.971 |

n = 60, so a refusal rate has a 95% CI of about ±10 points near 20% and about ±6 near 5%. The attention leg is the
exact hook; real C1 candidates leak more and refuse more.

## Conclusions and recommendation
1. **In-format, exact removal is impossible** (Q1). Only routing edits are available.
2. **Bias masking is dead.** It costs +0.19 KL, and with attn half the answers never arrive (31/60). This matches
   the literature.
3. **Conditional de-routing (`cond`, BF16 router rows) is the best in-format option.** Stacked on exact attention,
   refusal goes from 23% to 5% for +0.022 KL. The KL cost is large against the 0.02 target, so lower strength
   (d3) is enqueued as well.
4. **Router-row projection (`rproj`) is nearly free** (+0.002 KL, and 0.005 alone) but weak: 23% → 13% at K=8,
   which is inside the noise. K=32 is no better. Removing the router's sensitivity to h is not enough, because
   several top experts are selected on harmless text too and write refusal from their input.
5. **Owner decision (2026-11-21): drop-in only. The adapter is recorded here as a future option, not a
   deliverable.** The best effect per KL is an adapter, which is out of format. A rank-12 shared LoRA on the down_proj of the
   top-8 experts only, stacked on attn, gives 23% → 3% for +0.008 KL. Scoped to L8–11 on all experts it reaches
   0%, but for +0.045. This would change the deliverable to checkpoint + adapter code (a custom SGLang/Transformers
   patch to apply an expert-output LoRA). That is the owner's call.
6. **Weight candidates enqueued** (fleet, screen, priority 7), compact form, gate.weight only on top of the parent's
   wo_ud entries. build_cand.py builds all five on ink-research-3; the sha is in each sha256.txt.
   - `c4rp8_c1t002`, `c4rp8_c0lr`: rproj, K=8 (layers 9–11).
   - `c4cd6_c1t002`, `c4cd6_c0lr`: cond, K=32, d=6.
   - `c4cd3_c1t002`: cond, K=32, d=3.
   - **Risk:** the fleet's SGLang weight swap must accept `mlp.gate.weight`. It is in hf_name_map, but the SGLang
     router param name is unverified, so check the first screen's probe-swap log.
   - The `c0_lr` stacks (`c4rp8_c0lr`, `c4cd6_c0lr`) were enqueued before the lead said to skip c0_lr. Ignore
     their rows.
   - **Round 2 (14:45Z), built on the Mac:**
     - Stacked on mpoa's `c1h_x4_ouprofile` (exact4, r2, OU strength profile 4/2.5): `c4cd6_oupx4`,
       `c4cd3_oupx4`, `c4rp8_oupx4`.
     - Lower-KL cond points on c1t002: `c4cd6k8_c1t002` (top 8, layers 9–11) and `c4cd6k16_c1t002` (top 16).
     - The Mac builder uses a mini orig dir, `origdir/model.safetensors`: mpoa's range-fetched wo_ud and sconv
       tensors plus the 24 router weights, range-fetched at ed40e7da by `scripts/experts/fetch_gate_origdir.py`.
     - A Mac rebuild of `c4cd6_c1t002` matches the pod build bit-for-bit (sha 90f13e95…).
   - **Caveat:** the router bias is F32, which the candidate format does not allow (BF16 only), and hf_name_map
     has no gate.bias entry. The bias route is dead anyway.
