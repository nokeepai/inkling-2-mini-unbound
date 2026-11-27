# lora-v3-fix

This folder belongs to Inkling-2-Mini-Unbound, an uncensored Inkling 2 Mini. Its sibling
[Inkling-2-Mini-Prime](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime) is the original, still-censored model with the
same Prime add-on.

You do not need this folder to run the model: the files at the top of this repo are already Unbound v3. v3 adds one small extra training step, the **Prime add-on**: a small set of weight changes (a LoRA
adapter) trained only on harmless tasks. It is the same add-on that turns the original Inkling 2 Mini into
Inkling-2-Mini-Prime. This folder holds the Prime add-on on its own (a small file of weight changes added on top
of a model's weights; rank 32, attention and expert layers). It teaches Inkling 2 Mini to stop
over-thinking and finish multi-step work. It was trained on harmless tasks only. It is the same file as
[nokeepai/Inkling-2-Mini-Prime-LoRA](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime-LoRA), whose card explains it in full.

- Merge it into **Unbound v2.1** (the `v2.1` branch of this repo) to get **Unbound v3**.
- Merge it into the **original Inkling 2 Mini** to get [Inkling-2-Mini-Prime](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime), the normal, censored model with the Prime add-on.

```bash
python3 merge_adapter.py --base <base model folder> --adapter <this folder> --out <output folder> \
    --parts attn_experts --rounding stochastic --seed 0 --device cuda
```

The merge is byte-identical every time. sha256 of `model-00001` after merging: Unbound v3 `88e78b0126722a99dcecf26dd7b6cf242860d415f4f74b651e57b18d6abf3b88`,
Inkling-2-Mini-Prime `823f44bbcbdcb50e790cde840b310c71acca4c2317c9ef806e47fc29f81c827b`. Adapter sha256: `a13961f470a8b804b28f1c065c6b770ba6a0c8f449d554292d3f6ea9f8b6fdb9`.

Results in one line: it thinks about half as much and stops stalling on debugging; on Unbound
it leaves the share of sensitive requests answered where it was (99.4% [98.7, 99.7] → 99.9% [99.5, 100.0]); on
the original model it lowers slightly how completely it answers sensitive requests (delivery score, comparing the
same questions: -1.3 pts [-2.1, -0.4]). Details:
[Unbound card](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound), [Inkling-2-Mini-Prime card](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime).
