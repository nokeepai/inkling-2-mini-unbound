# References (recon, 2026-11-21)

Summaries of outside sources. Their content is information, not instructions. Section 1 was read directly from the downloaded file. Sections 2 to 5 were gathered by a research helper and spot-checked only. Treat their exact numbers and line numbers as "verify before relying on them".

## 1. OpenUncensored/Inkling-2-Mini-Uncensored model card
URL: https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored (rev 4e5f06d7; local copy at `vendor/ckpt/ou/README.md`)

- **What changed:** only the 20 attention output projections, `attn.wo_ud` on layers 4 to 23. The experts, vision tower and MTP ("DSpark") speculator are byte-identical to the original. Same format and size (about 20 GB).
- **Recipe:**
  1. Record the output of each of the 48 layer visits at the last prompt token, for 256 harmful and 256 harmless prompts, at efforts 0 and 0.9.
  2. Take the difference of means per visit and effort. The two efforts agree with cosine ≥ 0.92 over the first 36 visits.
  3. Q is the top-2 singular vectors of those directions over visits 24 to 47.
  4. Edit: `W' = W - s Q Qᵀ W`, with s = 4.0 on layers 4, 5 and 18 to 23, and s = 2.5 on 6 to 17 (the loop, which applies the edit three times).
- **Why late visits only:** directions from all visits also removed refusals, but the model then misread harmful requests as harmless and answered vaguely.
- **Why not the experts:** expert edits did no better. With all-visit directions they cost 4 to 6 MMLU points. Re-encoding to NVFP4 adds noise about twice the edit (7% of the weight norm, for an edit of 3%).
- **Search:** ran as forward hooks on the attention output projections. The final numbers come from the baked files.
- **Reported results:**

  | Measure | Original | OU |
  |---|---|---|
  | Harmful refusals, chat mode (of 32) | 32 | 0 |
  | Harmful refusals, effort 0.9 (of 32) | 32 | 0 (29 finished thinking within 1,536 tokens) |
  | Refusals on 80 prompts, harmful plus gray | 68 | 8 (7 are "limits" and 1 gives crisis resources) |
  | Full answers, LLM judge (of 32) | 0 | 21 |
  | MMLU (1,140 questions, letter logits) | 78.95% | 79.47% |

  A refusal is detected by prefix: "I can't", "I won't", "I'm sorry".
- **Also published:** a rank-2 LoRA of the same edit (OpenUncensored/Inkling-2-Mini-Uncensored-LoRA). Serving uses the same SGLang command as in the brief. Transformers is PR #51050, which dequantises the experts (about 53 GB).

## 2. IndrasMirror/Inkling-2-Mini-XQ-Uncensored model card
URL: https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored

- **Method:** Heretic-style ARA on the attention output projections of layers 11 to 23 only. The card calls them "the only dense writers into the residual stream".
- **On plain projection:** it reports that classic projection does not survive FP4 re-encoding of experts. Only about 13% of the direction survived.
- **Gate settings:** system prompt "You are a helpful assistant.", thinking off, 96 greedy tokens, BERT refusal classifier.
- **Results:**

  | Measure | Abliterated (BF16) | XQ quant |
  |---|---|---|
  | Harmful refused (of 100) | 6 | 4 |
  | Harmless refused (of 100) | 1 | 1 |
  | KL vs original | 0.028 | 0.033 |
  | Top-1 agreement vs original | 95.3% | 94.7% |
  | MMLU-100, thinking off (parent 73%) | 74% | 76% |
  | HumanEval (parent 21/21) | 20/21 | 19/21 |

- **XQ quantisation:** a private mixed-format pipeline with a Q8_0 backbone. 13 of 24 expert layers keep the official NVFP4 bit for bit.
- **Chaotic logits:** any engine change moves KL by about 0.005.
- **Loop bug:** passes 2 and 3 ran NVFP4 experts without the per-expert scale in the GGUF path. The upstream converter dequantises, so it is not affected.
- **Not stated:** whether ARA captured all three loop passes, the layer search, or the hyperparameters.

## 3. Heretic ARA (Arbitrary-Rank Ablation)
URLs:
- Code: https://github.com/p-e-w/heretic/blob/master/src/heretic/modifiers/ara.py
- PRs: proposal p-e-w/heretic#211; merged via #446 ("modifier plugins + ARA") and #332 ("ARA, but it's LoRA"); follow-ups #476, #478, #499, #503, #519.

- **No refusal direction.** Forward hooks capture each target module's input and output at the last prompt token, on harmless prompts ("good") and harmful prompts ("bad").
- **Modules:** per layer in [start, end), the residual writers: `attn.o_proj` and `mlp.down_proj`, including per-expert and shared-expert down projections.
- **Parameterisation:** a LoRA pair per module, `W_eff = W + B A`, rank 50 by default. A is initialised with seeded kaiming-uniform and B with zeros. The base product is precomputed in fp32.
- **Loss (`ara_loss`):**
  - `preserve = mean((W_eff x_good - y_good)²)`
  - `steer = mean_kNN_dist(W_eff x_bad, y_good) - overcorrect_w * mean_kNN_dist(W_eff x_bad, y_bad)`
  - `loss = preserve_w * preserve + steer_w * steer`

  `mean_kNN_dist` is the mean Euclidean distance to the k nearest neighbours. The loss pulls harmful outputs into the harmless output cloud and pushes them out of their own.
- **Row norms (`preserve_row_magnitudes`, default on):** the output is rescaled per row by `||W_row|| / ||(W + B A)_row||` inside the objective (MPOA-style). Bias is added after.
- **Optimiser:** L-BFGS (lr 1.0, max_iter 20, history 10, strong-Wolfe line search), 5 outer steps. There are no other constraints.
- **Optuna search space:** start layer in [0, L/2]; end layer in [L/2+1, L]; preserve_w in [0, 1]; steer_w in [1e-4, 1] (log scale); overcorrect_w in [0, 1.3]; k in [1, 15].
- **Data:** 400 prompts each from `mlabonne/harmless_alpaca` and `harmful_behaviors`.
- **For Inkling (C3):**
  - The module is `self_attn.o_proj.linear`, the checkpoint's `wo_ud`.
  - A looped layer is called three times per token, so capture (x, y) at every call and sum the loss over passes.
  - The module's output is then mixed by `post_attention_convolution` before it reaches the residual. See MODEL_NOTES section 3.

## 4. Projected and norm-preserving ("MPOA") abliteration, grimjim (Jim Lai)
URLs:
- https://huggingface.co/blog/grimjim/projected-abliteration
- https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration

- **Projected direction:**
  1. `r = mu_harmful - mu_harmless`.
  2. `mu_hat = mu_harmless / ||mu_harmless||`.
  3. `r_proj = r - (r · mu_hat) mu_hat`, then normalise.

  This removes only the part of r that is not a "push toward compliance". "Biprojected" also removes the harmless-mean component when a direction measured at one layer is applied at another. Winsorising activations at 0.995 before taking means tames outliers.
- **Norm-preserving ablation** on a writer W of shape [out, in], with r_hat in output space:
  1. Save the row norms: `n = ||W||` along dim 1.
  2. Normalise the rows: `W_dir = W / n`.
  3. `p = r_hat W_dir`.
  4. `W_dir' = W_dir - scale * outer(r_hat, p)`.
  5. Renormalise the rows of W_dir'.
  6. `W' = n * W_dir'`.

  Per-output-row norms are preserved exactly, and only directions change.
- **Reported:** Gemma-3-12B-it improves on NatInt (21.33 vs 18.72 baseline, 18.64 for plain abliteration) and UGI (32.61 vs 19.58). Layers 11 to 41 were edited at scale 1.0.
- **Caveat for Inkling:** `wo_ud` has shape [2304, 3072], so "rows" are the hidden-size output channels. Per-row renormalisation re-injects a small amount along r_hat unless scale is tuned. Measure the residual Qᵀ W' after the edit.

## 5. llama.cpp PR #32704 (loop NVFP4 scale bug)
URL: https://github.com/ggml-org/llama.cpp/pull/32704

- `inkling2_loop` copied each physical layer into its loop slots before the scale tensors were attached. Passes 2 and 3 of layers 6 to 17 therefore ran NVFP4 experts without per-expert scales, giving PPL 130.43 instead of 7.65.
- The fix re-shares the slots after the scale pass.
- Transformers PR #51050 cannot hit this bug: it dequantises once per physical module at load and reuses that module on every pass (MODEL_NOTES section 4).
