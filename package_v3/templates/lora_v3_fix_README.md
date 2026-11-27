# lora-v3-fix

This folder belongs to Inkling-2-Mini-Unbound, an uncensored Inkling 2 Mini. Its sibling
[{{names.prime_title}}](https://huggingface.co/{{names.prime_repo}}) is the original, still-censored model with the
same Prime add-on.

You do not need this folder to run the model: the files at the top of this repo are already Unbound v3. v3 adds one small extra training step, the **Prime add-on**: a small set of weight changes (a LoRA
adapter) trained only on harmless tasks. It is the same add-on that turns the original Inkling 2 Mini into
Inkling-2-Mini-Prime. This folder holds the Prime add-on on its own (a small file of weight changes added on top
of a model's weights; rank {{names.adapter_rank}}, attention and expert layers). It teaches Inkling 2 Mini to stop
over-thinking and finish multi-step work. It was trained on harmless tasks only. It is the same file as
[{{names.prime_lora_repo}}](https://huggingface.co/{{names.prime_lora_repo}}), whose card explains it in full.

- Merge it into **Unbound v2.1** (the `v2.1` branch of this repo) to get **Unbound v3**.
- Merge it into the **original Inkling 2 Mini** to get [{{names.prime_title}}](https://huggingface.co/{{names.prime_repo}}), the normal, censored model with the Prime add-on.

```bash
python3 {{files.merge_script}} --base <base model folder> --adapter <this folder> --out <output folder> \
    --parts attn_experts --rounding stochastic --seed 0 --device cuda
```

The merge is byte-identical every time. sha256 of `model-00001` after merging: Unbound v3 `{{files.unbound_sha}}`,
{{names.prime_title}} `{{files.prime_sha}}`. Adapter sha256: `{{files.lora_sha}}`.

Results in one line: it thinks {{headline.think_reduction}} as much and stops stalling on debugging; on Unbound
it leaves the share of sensitive requests answered where it was ({{ubench.willing_v21}} → {{ubench.willing_v3}}); on
the original model it lowers slightly how completely it answers sensitive requests (delivery score, comparing the
same questions: {{ubench.delivery_paired_prime}}). Details:
[Unbound card](https://huggingface.co/{{names.unbound_repo}}), [{{names.prime_title}} card](https://huggingface.co/{{names.prime_repo}}).
