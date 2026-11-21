# RULER long-context engine (agent suite/ruler, 2026-11-21)

Pod ink-suite-lc, 1x RTX PRO 6000 Blackwell Server Edition 96 GB. Same image (`lmsysorg/sglang:inkling-2-mini`,
id aff54906ae2e) and same serve command as harness/ENGINE.md via `scripts/suite/suite_serve.sh`; only the flag
overrides below differ. They are IDENTICAL for every model (orig, ou, finalists). Switch models by restarting the
container with another model dir; never change the flags.

```
bash /srv/suite/suite_serve.sh 0 3100 /w/<model> \
  --context-length 135168 --mem-fraction-static 0.85 --chunked-prefill-size 8192 \
  --max-running-requests 16 --random-seed 1234
```

| Flag | Value | Why |
|---|---|---|
| `--context-length` | 135168 (132k) | 131072-token RULER prompts plus chat template, answer prefix and up to 128 new tokens. Config max is 1M |
| `--mem-fraction-static` | 0.85 | gives 1,222,400 full-attention KV tokens + 122,240 SWA tokens (swa_full_tokens_ratio 0.1 default): about 9 concurrent 128k requests |
| `--chunked-prefill-size` | 8192 | long prompts prefill in 8k chunks; prefill CUDA graphs captured up to 8192 (7.8 GB) |
| `--max-running-requests` | 16 | client concurrency = 16 (RULER_CONC) |
| `--random-seed` | 1234 | project seed; greedy decoding does not use it |
| radix cache, CUDA graphs | default (on) | as ENGINE.md |

Not comparable with 5090 numbers (ENGINE.md: never compare across GPU types). All RULER numbers come from this one
GPU and these flags.
