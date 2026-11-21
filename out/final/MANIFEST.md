# Hash manifest

Original: `/root/w/orig` · Baked: `/root/pkg/out/c1h_r1_svd1_s3_L4` · 2026-11-21T17:08:24Z

**Verdict: PASS.** 1298 tensors hashed (sha256 of each tensor's bytes); 20 differ, 1278 are byte-identical. Safetensors headers (names, order, dtypes, shapes, offsets) are byte-identical.

## Edited tensors

| tensor | dtype | shape | sha256 original | sha256 baked |
|---|---|---|---|---|
| `model.llm.layers.10.attn.wo_ud.weight` | BF16 | [2304, 3072] | `cbc4ad684fda9b76…` | `59838935422b6aa4…` |
| `model.llm.layers.11.attn.wo_ud.weight` | BF16 | [2304, 3072] | `49c834a51d2c5e8a…` | `23308a907a2dcdae…` |
| `model.llm.layers.12.attn.wo_ud.weight` | BF16 | [2304, 3072] | `6434371abfbc7a2e…` | `adaf327d86417376…` |
| `model.llm.layers.13.attn.wo_ud.weight` | BF16 | [2304, 3072] | `92092cfd464ebb66…` | `af30fb7ef9a6f4ef…` |
| `model.llm.layers.14.attn.wo_ud.weight` | BF16 | [2304, 3072] | `8c162c53576546ec…` | `e36a37e525b81c99…` |
| `model.llm.layers.15.attn.wo_ud.weight` | BF16 | [2304, 3072] | `784c33a96c7d029a…` | `c8923dade6b6949f…` |
| `model.llm.layers.16.attn.wo_ud.weight` | BF16 | [2304, 3072] | `08f6012aeaa8dc30…` | `02797bffb285688c…` |
| `model.llm.layers.17.attn.wo_ud.weight` | BF16 | [2304, 3072] | `256633410371e4b5…` | `bae8940026bcdcf4…` |
| `model.llm.layers.18.attn.wo_ud.weight` | BF16 | [2304, 3072] | `ff991920db195e0b…` | `988c9b2fe33cb992…` |
| `model.llm.layers.19.attn.wo_ud.weight` | BF16 | [2304, 3072] | `6c15a1b3feb28a93…` | `13f5dfa80cf93762…` |
| `model.llm.layers.20.attn.wo_ud.weight` | BF16 | [2304, 3072] | `0461a4bc12a2f628…` | `08d45434fe673f54…` |
| `model.llm.layers.21.attn.wo_ud.weight` | BF16 | [2304, 3072] | `639d761035018b0b…` | `485c119244cd9157…` |
| `model.llm.layers.22.attn.wo_ud.weight` | BF16 | [2304, 3072] | `45157bbb79f4dc1d…` | `79347ed199e7661f…` |
| `model.llm.layers.23.attn.wo_ud.weight` | BF16 | [2304, 3072] | `3f4e5de3083c8602…` | `c6f8c92a2bcbafa8…` |
| `model.llm.layers.4.attn.wo_ud.weight` | BF16 | [2304, 3072] | `4bd504a9b75b880b…` | `ca001bf58f078df5…` |
| `model.llm.layers.5.attn.wo_ud.weight` | BF16 | [2304, 3072] | `63ee416934bfb09d…` | `3d0b083ed0dcaf19…` |
| `model.llm.layers.6.attn.wo_ud.weight` | BF16 | [2304, 3072] | `7e4dfcd9bc5282ad…` | `a89201902aa2cc55…` |
| `model.llm.layers.7.attn.wo_ud.weight` | BF16 | [2304, 3072] | `e658c3dfd46fb361…` | `7ac519736ac36ba0…` |
| `model.llm.layers.8.attn.wo_ud.weight` | BF16 | [2304, 3072] | `756866a408373439…` | `a012bc5ffd00b27d…` |
| `model.llm.layers.9.attn.wo_ud.weight` | BF16 | [2304, 3072] | `3dbfc13e09a8f317…` | `528c69441478e1c2…` |

## Files

| file | bytes | sha256 original | sha256 baked | same |
|---|---|---|---|---|
| `.gitattributes` | 1570 | `34448b82c17d60fe…` | `34448b82c17d60fe…` | yes |
| `README.md` | 16315 | `02f4cd9085aebde6…` | `c92f4ad65be51bd0…` | no |
| `chat_template.jinja` | 6868 | `53a9035f32559457…` | `53a9035f32559457…` | yes |
| `config.json` | 2722 | `a7a349601f0984c4…` | `a7a349601f0984c4…` | yes |
| `generation_config.json` | 98 | `a08737c07a65a8fd…` | `a08737c07a65a8fd…` | yes |
| `hf_quant_config.json` | 67924 | `7c656207d3ce3787…` | `7c656207d3ce3787…` | yes |
| `model-00001-of-00001.safetensors` | 19526019168 | `65e60f51aa1284d4…` | `438a092fa36e5e9c…` | no |
| `model.safetensors.index.json` | 109300 | `c1631065b7b00787…` | `c1631065b7b00787…` | yes |
| `mtp.safetensors` | 635801416 | `7c3b651317f01f73…` | `7c3b651317f01f73…` | yes |
| `processor_config.json` | 461 | `d7350adce18c8be4…` | `d7350adce18c8be4…` | yes |
| `special_tokens_map.json` | 529 | `c16f44b3594563d8…` | `c16f44b3594563d8…` | yes |
| `tiktoken/tokenizer.model` | 3386494 | `f170372dd6557289…` | `f170372dd6557289…` | yes |
| `tokenizer.json` | 26178678 | `7f332727f9a3f093…` | `7f332727f9a3f093…` | yes |
| `tokenizer_config.json` | 13235 | `bf09342d285139fd…` | `bf09342d285139fd…` | yes |

## Files added by the release (not in the original)

| file | bytes | sha256 |
|---|---|---|
| `abliteration/apply_edit.py` | 2149 | `6bb5ee604cc08abf…` |
| `abliteration/dora/adapter_config.json` | 1580 | `66035dadce1b1cf1…` |
| `abliteration/dora/adapter_model.safetensors` | 623640 | `7fee87a546e1a108…` |
| `abliteration/edit.pt` | 821297 | `3f24aa8fe6b1eede…` |
| `charts/banner.png` | 321745 | `ca99b2f24388291f…` |
| `charts/capability.png` | 147611 | `ab468b8c0d1d7f63…` |
| `charts/drift.png` | 88688 | `b206989f4146ef6e…` |
| `charts/hero.png` | 102402 | `f38381fdcf69c080…` |
| `charts/overrefusal.png` | 82890 | `c82b169a24892367…` |
| `charts/refusal.png` | 82060 | `eea6e7ead9c35fb0…` |
| `charts/useful.png` | 109763 | `35997531fca1e6e8…` |

Full per-tensor hashes for all tensors (identical and edited) are in MANIFEST.json.
