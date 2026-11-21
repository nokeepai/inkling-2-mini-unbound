[ink-abl] T+0:53  ALERT + APPROVAL REQUEST: Gate 0 criteria unreachable as written

Measured (methods/parity/PARITY.md):
- HF vs SGLang, identical first 64 greedy tokens: 2/50 (need ≥ 48/50)
- HF teacher-forced top-1 on SGLang tokens: 95.6% PRO 6000, 96.0% 5090 (need ≥ 99%)
- SGLang vs ITSELF (prefill vs decode, same GPU): 95.2% top-1, so 99% is unreachable for any engine
- SGLang 5090 vs SGLang PRO 6000: 3/50 exact, 95.3% top-1
- Loop passes 1/2/3 at layer 17: cosine 0.9975/0.9963/0.9870, smooth growth, no scale-bug jump → PASS
- Real engine difference: SGLang quantises expert activations (NVFP4) per the checkpoint config; HF keeps BF16 activations (5.4% relative error after layer 0)

Options:
A (recommended): recalibrate Gate 0 to the measured floor. HF vs SGLang top-1 ≥ 95% (≥ SGLang's own 95.2% minus 0.5), per-visit cosine ≥ 0.985 with no pass-boundary jump, coherent outputs. All generation metrics (refusal, judge, MMLU, GSM8K) are scored on SGLang on RTX 5090s only, for every model. KL and top-1 fidelity are computed only in HF, HF vs HF, per the brief. Cost: none.
B: first emulate NVFP4 activation quant in HF (~1 h of agent work, ~$3 of GPU), then re-measure. Even an exact emulation can't pass 99% because of the 95.2% self-floor.
C: stop per the brief's Gate 0 rule.

While waiting: Phase 2 diagnostics run in HF on the train split (unaffected). No SGLang scoring happens until you answer.
