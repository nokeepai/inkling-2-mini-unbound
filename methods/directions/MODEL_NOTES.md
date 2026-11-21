# Inkling-2-Mini model notes (recon, 2026-11-21)

## Sources and citation keys
- **Code:** Transformers PR #51050 at a10ebc3c, local copy in `vendor/transformers-pr51050/`.
- **Citation keys:**
  - **M** = `src/transformers/models/inkling2_loop/modeling_inkling2_loop.py`. This is the generated file, the one that actually runs.
  - **Mod** = `modular_inkling2_loop.py`
  - **C** = `configuration_inkling2_loop.py`
  - **CM** = `src/transformers/conversion_mapping.py`
  - **CL** = `src/transformers/core_model_loading.py`
  - **OC** = `utils/output_capturing.py`
- **Checkpoints:** small files of both checkpoints are in `vendor/ckpt/{orig,ou}/`. Tensor dtypes and shapes come from `hf_quant_config.json` and the safetensors header on ink-research-1.
- **HF module paths** below are for `Inkling2LoopForConditionalGeneration` (the checkpoint's `architectures`). Prefix `P = model.model.language_model`.
  - For `Inkling2LoopForCausalLM`, P is `model.model`.

## 1. Layer stack, loop and visit indexing

### Shape
- 24 physical decoder layers, hidden size 2304, vocab 201024. The LM head is unpadded to 200058 rows on load (CL:856-878).
- **Attention type:** full ("hybrid") on layers 5, 11, 17 and 23; sliding window 1024 on every other layer (C:150-153, matching `local_layer_ids` in config.json).
- **Heads:** 24 q heads and 8 kv heads, head_dim 128, in both layer types.
- **Loop:** layers 6 to 17 (`core_start=6`, `core_end=18`) run `num_passes=3` times. `total_num_hidden_layers = 24 + 2*12 = 48` (C:173).

### Forward order (M:1308-1446)
- Pre-loop: layers 0 to 5 (M:1362).
- Save `pre_loop_hidden_states` and `pre_loop_block_states` (M:1378).
- Pass 1: layers 6 to 17 with no messages (M:1381).
- Passes 2 and 3 **restart from the pre-loop state** and receive messages built from the previous pass (M:1397-1424).
- Post-loop: layers 18 to 23 (M:1426).
- Then `final_input` (block attention over all blocks), `norm`, `lm_head`, and a logit softcap of 30 (M:1442-1443, M:1815-1816).

### Visit index v (0..47, in call order)

| v | layer | pass |
|---|---|---|
| 0-5 | 0-5 | n/a |
| 6-17 | 6-17 | 1 |
| 18-29 | 6-17 | 2 |
| 30-41 | 6-17 | 3 |
| 42-47 | 18-23 | n/a |

- In general, v = layer for v < 18, then `6 + (v-6) % 12` through v = 41, and v - 24 for v ≥ 42.
- **What OU's "visits 24-47" covers under this indexing:** pass 2 layers 12-17, all of pass 3, and post-loop layers 18-23.
- **Ambiguity:** OU does not define its indexing. Under the KV-slot order below, "24-47" would instead mean passes 2 and 3 only. Phase 2 must pick one indexing and state it.

### KV and conv cache slots
- The cache gets 12 deep-copied loop layers appended per extra pass (M:1354-1357). Every visit owns its own keys, values and conv states (docs `inkling2_loop.md`:182).
- Slot = `layer_idx + cache_offset`, where `cache_offset = 24 + (pass-2)*12 - 6` for passes 2 and 3, and 0 otherwise (M:1405).
- Resulting map:
  - Layers 0-23 at pass 1, and the non-loop layers: slots 0-23.
  - Pass 2, layers 6-17: slots 24-35.
  - Pass 3, layers 6-17: slots 36-47.
- **Slot ≠ visit index** for post-loop layers: layer 18 is visit 42 but slot 18.

### Why passes 1 and 2 only matter through messages
- Passes 1 and 2 reach the rest of the network **only through messages** and through their own KV slots, which later tokens in the same pass read.
- After pass 3, block_states holds `[emb, B0-5, B6-11(pass 3)]`, plus `B12-17(pass 3)` pushed at layer 18. Post-loop layers and `final_input` see only pass-3 blocks.

## 2. Residual stream: block attention residuals, not a plain sum
This differs from a standard transformer and matters for "the residual state".

### Blocks
- `n_layers_per_attn_res_block = 6`. Layers 0, 6, 12 and 18 are block boundaries (M:861).
- Within a block, `hidden_states` is a running partial sum: `out = block_residual + attn_write + mlp_write` (M:905, 910).
- **At a boundary layer the sum restarts:** `out = attn_write + mlp_write`, and the incoming `hidden_states` is pushed into `block_states` (M:902-903; text model M:1374-1375).

### What each sublayer reads
- Each sublayer reads a learned attention over the RMS-normalised blocks: `Inkling2LoopBlockAttentionResidual`, with a fixed learned query and softmax over `[block_states..., partial]` (M:828-853).
  - `attention_input`: 1 query (3 queries for full-attention layers: q, k, v); absent on layer 0.
  - `mlp_input`: 1 query.
  - `final_input` before the final norm.
- **Values are RMS-normalised** (`attn_res_normalize_values: true`). Readers therefore see a convex mix of unit-RMS block states, and the scale of a block does not matter, only its direction.
- **Implication for directions:**
  - A decoder layer's output (what `output_hidden_states` records) is a *partial block sum*.
  - At v = 0, 6, 12, 18, 30 and 42 (boundaries) it contains only that layer's own writes.
  - Difference-of-means at those visits are not comparable to their neighbours.
  - Recommendation: also record the `mlp_input` output (what the MoE actually reads) at each visit.

### Exact writers into the partial sum, per layer (M:876-910, 937-997, 1022-1062, 1091-1149)
- **Attention write:** `post_attention_convolution(o_proj(attn, gate))`.
  - `o_proj` = `sigmoid(gate_proj(x)) * attn`, then `linear` (M:531-543).
  - `post_attention_convolution` is a depthwise causal conv over tokens, kernel 4, then `(conv(h) + h) / sqrt(5)` (M:329-393, line 392). Its weights are kept in fp32 (M:1192-1204).
- **MoE write:** `post_mlp_convolution(experts(...))`, the same conv form.
- **Loop message gates change only sublayer inputs, never the sum.**
  - `q/k/v_message_gate` act on the attention inputs.
  - `mlp_message_gate` acts on the MLP input after `mlp_input` (M:965-970, 992, 1117-1119, 1144).
  - `residual` is taken before them.

## 3. Module paths

| What | Checkpoint tensor (model.llm.…) | HF module (under P) | dtype in ckpt |
|---|---|---|---|
| Embedding | `embed.weight` [201024,2304], `embed_norm.weight` | `embed_tokens` (Embedding + RMSNorm, M:1240-1246) | BF16 |
| Attn output proj | `layers.i.attn.wo_ud.weight` [2304,3072] | `layers[i].self_attn.o_proj.linear` (CM:394) | BF16 |
| Attn output gate | `layers.i.attn.gate_attn.weight` [3072,2304] | `layers[i].self_attn.o_proj.gate_proj` (CM:395) | BF16 |
| Post-attn conv (the actual write) | `layers.i.attn_sconv.weight` [2304,1,4] | `layers[i].post_attention_convolution` (CM:397) | BF16 → fp32 |
| q/k/v proj | `attn.wq_du` [3072,2304], `wk_dv`/`wv_dv` [1024,2304] | `self_attn.{q,k,v}_proj.linear` | BF16 |
| k/v short conv | `attn.{k,v}_sconv.weight` (sliding) or `static_w` + `weight_proj` (full) | `self_attn.{k,v}_proj.sconv` | BF16 |
| Sinks, rel-pos | `attn.attn_sink_proj`, `attn.wr_du`, `rel_logits_proj.proj` | `self_attn.sink_proj`, `self_attn.relative_position_bias.*` | BF16 |
| Norms | `attn_norm` (sliding), `q/k/v_input_norm` (full: 5, 11, 17, 23), `mlp_norm` | `input_layernorm` or `{q,k,v}_input_layernorm`, `post_attention_layernorm` | BF16 |
| Router | `layers.i.mlp.gate.weight` [128,2304], `gate.bias` | `layers[i].mlp.gate` (sigmoid scores, top-8, bias-corrected, ×8 route_scale, M:1152-1172) | BF16 weight (fp32 in model), F32 bias |
| Experts (MoE block) | `mlp.experts.w13_weight`, `w2_weight` (+ scales) | `layers[i].mlp.experts.{gate_up_proj [128,2304,2304], down_proj [128,2304,1152]}`; block = `layers[i].mlp` | NVFP4 (L0-19) / MXFP8 (L20-23) |
| Post-MoE conv (the actual write) | `layers.i.mlp_sconv.weight` | `layers[i].post_mlp_convolution` | BF16 → fp32 |
| Shared experts | none (`n_shared_experts: 0`) | none | n/a |
| Block-residual queries | `attn_res.attn_queries.N` (offset +1: N → layer N+1), `attn_res.mlp_queries.N`, `attn_res.output_query` | `layers[i].attention_input.query`, `layers[i].mlp_input.query`, `final_input.query` | F32 |
| Loop controllers | `layers.i.controllers.{attn,k,v,mlp}.{down_weight [64,4608], forget_weight [2304,64], gain_weight [2304,64]}`, i = 6..17 | `layers[i].{q,k,v,mlp}_message_gate.{down_proj, forget_gate_proj, message_gate_proj}` (CM:2685-2700) | BF16 |
| Message queries | `middle_cycle_cross_pass_queries` [24,2304], `…_kv_queries` [24,2304] | `message_input.query` [12,4,2304] (q, k, v, mlp per loop layer; CL:823-849) | F32 |
| Final norm | `norm.weight` | `norm` | BF16 |
| LM head | `unembed.weight` [201024,2304], untied | `model.lm_head` (M:1732), plus `logit_softcap` 30 | BF16 (fp32 output) |
| Vision tower | `model.visual.*` (23 layers, hidden 1024) | `model.model.vision_tower` (M:1578) | BF16 |
| MTP speculator | `mtp.safetensors` (separate file) | not loaded by the PR model | BF16 |

## 4. Precision, and how the PR dequantises

### dtype groups (`hf_quant_config.json`)
- **F32:** attn_res queries, router bias, message queries (74 tensors).
- **BF16:** everything else that is dense (993 tensors, including the vision tower), plus unembed (fwd output fp32).
- **NVFP4:** experts `w13`/`w2` on layers 0-19. Codes are U8, with an F8_E4M3 scale per 16 values, a per-expert F32 `amax`, and an F32 `outer_scale_denominator`.
- **MXFP8:** experts on layers 20-23. F8_E4M3 values with a U8 (ue8m0) scale per 32 values.
- Header totals: F32 154, BF16 994, F8_E4M3 48, U8 48.

### Which BF16 tensors write into the residual partial sum
- `attn.wo_ud` (through `attn_sconv`), `attn_sconv`, `mlp_sconv`, and the embedding (`embed` + `embed_norm`, block 0).

### "attn wo_ud is the only dense writer": partly true
- **True part:** `wo_ud` is the only dense *matrix* whose output is added to the residual. The other writer is the MoE, which is quantised.
- **Caveat 1, the conv:** its output passes through `post_attention_convolution` before it is added.
  - The conv computes `(h_t + Σ_k diag(w_k) h_{t-k}) / √5` with per-channel taps.
  - So `Qᵀ W' = 0` (a perfect projection on `wo_ud`) does **not** make the written vector orthogonal to Q.
  - Per-channel taps mix the remaining components back into span(Q).
  - The leak is set by how unequal the taps `w_k` are across channels. It is measurable, but not yet measured.
  - Hooks on `post_attention_convolution` output remove Q exactly. Hooks on `o_proj` (equal to a weight edit) do not.
- **Caveat 2, the gate:** `gate_attn` (BF16) multiplies the attention output before `wo_ud`.

### Dequantisation (CL:772-816, wired at CM:2654-2677)
- Expert weights are dequantised to the param dtype once at load, 8 experts per chunk.
  - NVFP4: E2M1 lookup, times the fp8 block scale, times `amax / outer_scale_denominator`.
  - MXFP8: value times `2**(scale-127)`.
- **Size:** experts are 24 × 128 × (2·1152·2304 + 2304·1152) ≈ 24.5 B params, which is 48.9 GB in BF16. With embed, unembed (0.93 B) and the dense rest, that is about 53 GB, matching the brief.
- **Loop passes share scales:** passes 2 and 3 call the very same `layers[i]` module objects (M:1406-1419). They use the same dequantised tensors and the same scales by construction. The llama.cpp per-slot scale bug cannot occur here.
- **Parity caveat:** `hf_quant_config` declares `required_activation_quantization` for the experts.
  - NVFP4 layers: per-token NVFP4 activations ("fouroversix_mae").
  - MXFP8 layers: MXFP8 activations.
  - The PR runs weight-only dequant with BF16 activations. If SGLang quantises activations, small Transformers-vs-SGLang differences are expected, and the parity gate should allow for them.

## 5. Loop controllers: per layer, shared across passes 2 and 3
**Answer: NOT per pass.** Each looped layer i (6..17) has four controllers (`q`, `k`, `v`, `mlp` message gates), each low-rank (rank 64). One set of weights is used on both pass 2 and pass 3; pass 1 uses none.

### Gate (M:79-96)
```python
gate_hidden_states = torch.cat([self.norm(hidden_states), self.norm(message)], dim=-1)   # 4608
gate_hidden_states = self.down_proj(gate_hidden_states)                                     # -> 64
forget_gate = torch.tanh(self.forget_gate_proj(gate_hidden_states))                         # -> 2304
message_gate = torch.tanh(self.message_gate_proj(gate_hidden_states))
return (1 - forget_gate) * hidden_states + message_gate * message
```

### Where it is applied (M:1397-1419)
```python
for loop_idx in range(1, self.config.loop_num_passes):          # passes 2 and 3
    previous_block_states = block_states[..., num_pre_loop_blocks + 1 :, :]
    previous_hidden_states = hidden_states
    hidden_states, block_states = pre_loop_hidden_states, pre_loop_block_states
    cache_offset = self.config.num_hidden_layers + (loop_idx - 1) * num_loop_layers - loop_start
    for loop_layer_idx, decoder_layer in enumerate(self.layers[loop_start:loop_end]):   # same modules every pass
        messages = self.message_input(previous_block_states, previous_hidden_states, loop_layer_idx)
        new_hidden_states = decoder_layer(hidden_states, block_states, messages, ...)
```
- The messages come from `message_input`, with query [12,4,2304] indexed by `loop_layer_idx` only, not by `loop_idx` (M:145-173).
- They attend over the previous pass's `[B6-11, partial B12-17]`.
- **Inside the layer** (sliding: M:960-970, 992; full: M:1114-1119, 1144):
  - Sliding layers apply the q/k/v gates and then `input_layernorm`.
  - Full layers apply the gates after `attention_input` and before `{q,k,v}_input_layernorm`.
  - The MLP gate is applied after `mlp_input` and before `post_attention_layernorm`.
- Pass 1 has `messages=None`, so the controllers are skipped.
- **Consequences:**
  - C5 ("pass-specific edit in loop controllers") has no pass-specific weights to edit. A pass-specific edit would need a hook that branches on pass index, which can't be baked into one weight file.
  - The only per-pass difference in weights is none. Per-pass behaviour comes only from the messages and from the separate KV slots.

## 6. Chat template, thinking and effort
- **Source:** `chat_template.jinja`, identical in orig and OU. Rendered with transformers 4.57 and the checkpoint tokenizer; the script is `scripts/recon/render_template.py`.

### Effort
- Effort is a **system message**, `Thinking effort level: <x>`, controlled by the template kwarg `reasoning_effort`.
- It accepts a float in [0, 0.99] or one of the names: none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh/max 0.99.
- Default when unset: 0.9. "Thinking off" means effort 0, rendered as `0`.
- It is emitted once, after any leading system messages and immediately before the first non-system message.

### Template structure
- **No BOS** is added, and `add_special_tokens` changes nothing.
- **No generation prompt:** `add_generation_prompt` is ignored, so the prompt ends at the user's `<|end_message|>`.
- **The model generates** `<|message_model|>`(200001) `<|content_thinking|>`(200008) … `<|end_message|>`(200010), then `<|message_model|><|content_text|>`(200004) answer `<|end_message|><|content_model_end_sampling|>`(200006 = eos).
- **Parser:** `response_template` in `tokenizer_config.json` splits `reasoning_content` (after `<|content_thinking|>`) from `content` (after `<|content_text|>`). Each runs up to `<|end_message|>`.
- **Effort 0 still thinks:** the README says "minimum effort does not fully disable reasoning", so even at effort 0 the first tokens may be `<|content_thinking|>`.

### Rendered examples (user "How do I bake bread?")
- Thinking off (effort 0), 20 tokens:
  `<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>How do I bake bread?<|end_message|>`
- Effort 0.5, 22 tokens:
  `<|message_system|><|content_text|>Thinking effort level: 0.5<|end_message|><|message_user|><|content_text|>How do I bake bread?<|end_message|>`
- Effort 0.9 (also the default), 22 tokens:
  `<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>How do I bake bread?<|end_message|>`
- Token ids: prompt start `200002 200004 …`; end of user instruction `… 30('?') 200010`.
- With a system prompt, the order is: system, then the effort system message, then the user.

### Positions for Phase 2
- **Last instruction token:** the token before the final `200010`.
- **Template tokens after it:** only `<|end_message|>` (200010) is in the prompt.
  - To probe "post-instruction template" positions, append the model's forced prefix `<|message_model|><|content_thinking|>` (200001 200008).
  - "First thinking token" = position of 200008, the last token of the forced prefix, whose output predicts the first thinking word.
  - Treat the forced prefix as standard. It is exactly what the model emits, per the docs examples (`inkling2_loop.md`:66, 112).

## 7. Hooks for Phase 2
1. **Residual state per visit:**
   - Easiest: `model(..., output_hidden_states=True)`. `hidden_states` has 49 entries: [0] is the embeddings and [v+1] is visit v (OC:138-145; recorder M:1205-1214 with `total_num_hidden_layers`).
   - **Caveat:** the last entry is overwritten with `last_hidden_state`, which is after `final_input` and `norm` (OC:362-375). Visit 47's raw output therefore needs a hook.
   - Hook form: `register_forward_hook` on each `P.layers[i]`, with a per-module call counter reset by a `forward_pre_hook` on `P`.
   - For i in 6..17 the k-th call within one forward is pass k. Global call order equals visit order.
2. **What the layer reads:** forward hook on `P.layers[i].mlp_input` and `P.layers[i].attention_input`. Output is a tuple; take [0], or [0..2] for full layers.
3. **Attention write per layer and pass:**
   - Pre-conv: hook `P.layers[i].self_attn.o_proj` or `.o_proj.linear`; the latter is equivalent to weight edits.
   - Post-conv (the true residual write): hook `P.layers[i].post_attention_convolution`.
4. **MoE write per layer and pass:**
   - Pre-conv: `P.layers[i].mlp` (SparseMoeBlock output).
   - True write: `P.layers[i].post_mlp_convolution`.
   - Router logits: `P.layers[i].mlp.gate` output[0].
5. **Pass index:** the same call-counter scheme applies to every submodule of looped layers (3 calls per forward).
   - With a KV cache during generation, each decode step is one forward, so reset the counters per forward.
   - The cache slot used is `layer + {0, 18, 30}[pass-1]`.
6. **Ablation hooks:** an output hook returning `y - y @ Q @ Q.T` on `o_proj.linear` reproduces a weight edit with s = 1. OU's s > 1 is `y - s * y @ Q @ Q.T`.

## 8. OpenUncensored edit.pt
- **Summary:** `methods/directions/ou_edit_summary.json`. Script: `scripts/recon/edit_summary.py`, loaded with `weights_only=True` on CPU.
- **Keys:**
  - `name` = `'qlate2_out4.0_core2.5_attn'`
  - `Q`: float32 [2304, 2]. Orthonormal, rank 2; singular values 1.0, 1.0; `QᵀQ` = I within 1e-6.
  - `s_attn`: float32 [24]
  - `s_mlp`: float32 [24], all zero
  - `note` (string)
- **Strengths:** `s_attn` = 0 on layers 0-3, 4.0 on 4-5, 2.5 on 6-17, and 4.0 on 18-23. 20 tensors are edited, matching the OU README.
- **Formula, per the note and README:** `W' = W − s · Q Qᵀ W` on `model.llm.layers.i.attn.wo_ud.weight` [2304, 3072].
  - On span(Q) this multiplies the output by 1 − s: **−3 outside the loop and −1.5 in the loop**.
  - It is not a projection. It **reflects and amplifies** the Q component. OU actively writes the negative refusal direction.
  - Looped layers apply it once per pass. Each of the 3 visits writes with eigenvalue −1.5 on span(Q), but into separate pass states, and only pass 3's blocks reach the post-loop layers directly.
  - "Three times per token" is true per weight application, but not as a compounding of the same residual.
- **Q is in output (hidden, 2304) space.** Rows of `wo_ud` index the hidden dims. Q's largest entries are at dims 1915, 478 and 1206 (col 0) and 2012, 548 and 940 (col 1). The largest |entry| is 0.17, so Q is not concentrated on a few channels.

## 9. OU edit verification
Run on ink-research-1, CPU only, at about 09:45 UTC.
- Scripts: `scripts/recon/verify_ou_edit.py` and `scripts/recon/formula_variants.py`.
- Raw results: `methods/directions/recon_verify.json` (on the pod: `/root/logs/recon_verify.json` and `/root/logs/recon_variants.json`).

### Which tensors differ
- Both files have 1244 tensors, with the same names, dtypes and shapes, and the same `__metadata__` (`{'format':'pt'}`).
- By per-tensor blake2b hash, **exactly 20 tensors differ**: `model.llm.layers.{4..23}.attn.wo_ud.weight`.
- Everything else is byte-identical: all experts and scales, the controllers, embed/unembed and the vision tower. `mtp.safetensors` was not hashed, but its size is identical.

### The formula reproduces OU bit for bit
Recipe:
```python
W32 = W_bf16.float(); Q32 = edit["Q"]            # float32
W_new = (W32 - s * (Q32 @ (Q32.T @ W32))).to(torch.bfloat16)
```
- **Exact match:** 0 differing elements out of 7,077,888 on layers 4 (s=4), 6, 12 and 17 (s=2.5).
- **Operation order matters at the BF16 ULP level for s = 2.5:**

  | Variant | Differing elements |
  |---|---|
  | `(s*Q) @ (Qᵀ W)` | 21–34 per tensor, max \|diff\| 2.4e-4 |
  | `(Q Qᵀ) W` | ~500 |
  | `(I − sQQᵀ) W` | ~700 |
  | fp64 | ~500 |

- For s = 4 (a power of two), the `(s*Q)` order is also exact on all 8 tensors.
- **Conclusion:** use the recipe above for C0. All 20 edited tensors match with this recipe, 4 checked bit-exact; the other 16 are within 1 BF16 ULP on ≤ 34 elements under the alternate order.

### What the edit does to the weights
- **Row norms:** the original `wo_ud` rows (one per hidden dim) are **unit-norm**, at 0.9991–1.0004 on every layer checked, with Frobenius norm exactly 48 = √2304 on all 24 layers. This looks like a training-time constraint.
- **OU breaks this:** row norms become 0.99–1.11 at s=4 and 0.99–1.05 at s=2.5, a max relative change of 9–15% at s=4 and 1.5–5% at s=2.5. A norm-preserving variant (C1) restores the original's actual invariant, not just a heuristic.
- **‖QᵀW‖** goes from about 1.4 to about 4.2 at s=4 (×3) and to about 2.2 at s=2.5 (×1.5). This confirms the reflection (eigenvalue 1−s) rather than removal.
- **The delta lies in span(Q):** fraction 0.9996–1.0, with the remainder from BF16 rounding.

## 10. Checks against BRIEF section 9
- **24 layers; 6-17 loop ×3; 48 visits; 48 cache slots.** Confirmed. Note that the slot order differs from visit order (section 1).
- **"Passes 2 and 3 read the previous pass through per-layer low-rank loop controllers."** Confirmed: rank 64, per layer, four gates (q, k, v, mlp). They are **shared by passes 2 and 3**, not per pass, so C5's precondition is false. Messages come from an attention over the previous pass's blocks, using per-loop-layer queries.
- **"Layers 6-17 run three times with shared weights" with "an edit to a looped weight acts three times".** True per application. But passes restart from the pre-loop state, and only pass 3's blocks feed post-loop layers. Passes 1 and 2 matter only via messages and their own KV slots, so the three applications do not stack additively on one residual.
- **Experts: NVFP4 on layers 0-19, MXFP8 on 20-23, 128 experts, top-8.** Confirmed. There are no shared experts.
- **"attn.wo_ud is the only dense writer."** Partly true. It is the only dense matrix writer, but its output goes through a per-channel causal conv (`attn_sconv`) before entering the residual. Projecting out of `wo_ud` therefore leaves some leak of Q into the residual (section 4).
- **The residual stream is not a running sum** (new, not in the brief). It uses 6-layer blocks with learned, RMS-normalised attention over blocks, and the per-layer output restarts at layers 0, 6, 12 and 18 (section 2).
- **OU recipe.** Confirmed and bit-exact (section 9). Note that s > 1 makes it a reflection, not an ablation.
- **"Always thinks."** Consistent with the template and README: effort 0 is only a system message ("Thinking effort level: 0"), not a mode switch.
- **Transformers dequant, about 53 GB, loop scales.** Confirmed. Passes reuse the same modules, so the llama.cpp bug cannot occur. Separately, the checkpoint declares NVFP4/MXFP8 *activation* quantisation for the experts, which the PR does not emulate; this is a possible source of SGLang-vs-HF parity drift.

### Unknowns
- **Exact leak of Q through `attn_sconv`.** Needs the conv weights and a quick measurement. Hook `post_attention_convolution` with and without the edit.
- **Whether SGLang applies activation quantisation to experts.** Read the SGLang image's inkling model code, or compare per-pass hidden states (Phase 0 item 4).
- **Visit indexing.** It is not clear which indexing OU meant by "visits 24-47" (call order or KV-slot order). Their README says "the output of each of the 48 layer visits", which suggests call order (section 1).
