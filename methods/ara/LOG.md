# ARA (C3, loop-aware) log

Code: R/scripts/ara/ (ara_capture.py = pod capture, ara_fit.py = per-layer fit, fit_job.sh + verify_build.py = pod build check, search.py = Optuna loop). Study: state/optuna/c3.db.

## Setup (12:18Z)
- Capture (train split only): 256 harmful + 256 harmless at effort 0 and 0.9 each (1024 prompts). Prompt + forced `<|message_model|><|content_thinking|>` + 8 own greedy thinking tokens, teacher-forced.
- Per layer L4-23 and every call (passes 1-3 for L6-17): wo_ud input x on a 15-token window = 3 conv-context tokens + 12 targets (last instr, end_msg, msg_model, content_thinking, thinking 1-8).
- Post-conv formula y_t = ((1+w3)h_t + w2 h_{t-1} + w1 h_{t-2} + w0 h_{t-3})/sqrt5 reproduces the captured post_attention_convolution output to 0.2-0.3% rel. error (BF16) on L4/6/11/17/23, all passes.
- Edit W' = diag(r)(W + A Bᵀ), r = row-norm preservation (exact, in the objective). Loss on post-conv writes, averaged over (effort, pass, position) groups: preserve_w * ||y'_good - y_good||² + kNN(y'_bad -> good cloud) - overcorrect * kNN(y'_bad -> bad cloud) + mean_w * ||mean y'_bad - (mu_good + gamma (mu_good - mu_bad))||² (gamma > 0 = overshoot, to make attention compensate for the MoE's refusal write). Adam, A = 0 init.
- Compact form ships A, B, r directly; build_cand reproduces W' bit-exactly (checked on the pod per candidate: verify.json exact_cpu = n).

## Candidates
- 12:30Z ink-research-2 (kl_worker) went away; worker errors now re-enqueue instead of failing trials.
- 12:45Z proxy sweep started (PROXY_LOG.md, train held-out, diagnostic). Finding: every config sits at proxy KL ~0.03 once the refusal projection at L23 reaches the harmless level; preserve_w on the 12 decision positions does not lower it.
- 13:04Z added a broad preserve set (P{l}.pt: 384 held-out harmless prompts, own thinking/reply tokens 8-39, all passes) and term pres2_w. Same config: proxy KL 0.032 -> 0.0155 (pres2_w 10) / 0.0135 (100) at the same refusal score. pres2_w added to the c3 space (log 1-300); c3t000-c3t008 predate it.
- 14:20Z c3x001 = pod job r3t001 (L4-23 k16 pw10 pres2 30 mean 1 γ0.5) and c3x002 = r3t002 (L11-23 pres2 30 mean 2 γ1 lr5e-4), promoted by hand (not Optuna trials).
- 14:48Z c3x001 [full] kl=0.0304 top1=0.9389 refusal_off=0.0285 refusal_e09=0.0285 mmlu300=76.67 gsm8k100=97 e09_rep=0.544 e09_cap=0.238 honest=0.9 overrefusal=0.005 (fails rep/cap/honest)
- 14:25Z c3x002 [screen] kl=0.0263 top1=0.9405 screen_refusal_off=0.0917 mmlu300=76
- 12:27Z study reset (first try failed on the scp download flag order; no candidate was enqueued).
- 12:30:16Z c3t001 [fail] {"l_min":4,"l_max":23,"rank":32,"preserve_w":1.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":0.0,"gamma":0.0} fit[dW=0.1419 gap=0.6617] -> 
- 14:09:42Z c3t010 [screen] {"l_min":11,"l_max":23,"rank":16,"preserve_w":10.0,"pres2_w":100.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":1.0,"gamma":0.5} -> kl=0.01275 top1=0.9607 screen_refusal_off=0.5417 screen_mmlu300=76
- 14:09:42Z c3t013 [screen] {"l_min":11,"l_max":23,"rank":16,"preserve_w":10.0,"pres2_w":300.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":1.0,"gamma":0.5} -> kl=0.01156 top1=0.9622 screen_refusal_off=0.6417 screen_mmlu300=76.33
- 14:09:42Z c3t014 [screen] {"l_min":11,"l_max":23,"rank":16,"preserve_w":10.0,"pres2_w":300.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":0.5,"gamma":0.3} -> kl=0.007764 top1=0.9698 screen_refusal_off=0.725 screen_mmlu300=77
- 14:18:49Z c3t000 [screen] {"l_min":11,"l_max":23,"rank":16,"preserve_w":1.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":0.0,"gamma":0.0} -> kl=0.04609 top1=0.9229 screen_refusal_off=0.3833 screen_mmlu300=76.33
- 14:18:49Z c3t005 [screen] {"l_min":11,"l_max":23,"rank":16,"preserve_w":10.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":0.0,"gamma":0.0} -> kl=0.02216 top1=0.9482 screen_refusal_off=0.4417 screen_mmlu300=78.33
- 14:19:19Z c3t004 [screen] {"l_min":4,"l_max":23,"rank":32,"preserve_w":1.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":0.0,"gamma":0.0} -> kl=0.108 top1=0.8876 screen_refusal_off=0 screen_mmlu300=75.33
- 14:19:34Z c3t003 [screen] {"l_min":11,"l_max":23,"rank":16,"preserve_w":1.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":1.0,"gamma":1.0} -> kl=0.1271 top1=0.8733 screen_refusal_off=0.008333 screen_mmlu300=74.33
- 14:19:34Z c3t007 [screen] {"l_min":7,"l_max":23,"rank":16,"preserve_w":17.78902,"overcorrect":0.06864,"knn_k":2,"steps":263,"lr":4e-05,"think_w":0.91919,"mean_w":2.00568,"gamma":0.85834} -> kl=0.1417 top1=0.8793 screen_refusal_off=0 screen_mmlu300=73.67
- 14:20:20Z c3t008 [screen] {"l_min":7,"l_max":23,"rank":16,"preserve_w":17.78902,"overcorrect":0.06864,"knn_k":2,"steps":263,"lr":4e-05,"think_w":0.91919,"mean_w":2.00568,"gamma":0.85834} -> kl=0.1417 top1=0.8793 screen_refusal_off=0 screen_mmlu300=74.67
- 14:31:11Z c3t011 [screen] {"l_min":11,"l_max":23,"rank":16,"preserve_w":10.0,"pres2_w":10.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":1.0,"gamma":0.5} -> kl=0.01871 top1=0.9506 screen_refusal_off=0.275 screen_mmlu300=76.33
- 14:31:41Z c3t012 [screen] {"l_min":6,"l_max":20,"rank":32,"preserve_w":0.46037,"pres2_w":10.20307,"overcorrect":0.82599,"knn_k":8,"steps":181,"lr":0.0001,"think_w":1.95533,"mean_w":0.55291,"gamma":0.48454} -> kl=0.0599 top1=0.9168 screen_refusal_off=0.01667 screen_mmlu300=77
- 14:46:28Z c3t009 [full] {"l_min":7,"l_max":23,"rank":16,"preserve_w":17.78902,"pres2_w":1.4792,"overcorrect":0.07533,"knn_k":9,"steps":126,"lr":0.00011,"think_w":1.41998,"mean_w":1.28751,"gamma":0.08791} -> kl=0.03665 top1=0.9371 screen_refusal_off=0.01667 screen_mmlu300=75.67 refusal_off=0.02135 refusal_e09=0.01068 mmlu300=74.33 gsm8k100=98 harmful_e09_cap_hit=0.1495 harmful_e09_rep=0.3772
- 15:42:20Z c3t015 [fail] {"l_min":11,"l_max":23,"rank":16,"preserve_w":10.0,"pres2_w":300.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":1.5,"gamma":1.0} -> kl=0.01876 top1=0.9516
- 15:44:35Z c3t016 [fail] {"l_min":11,"l_max":23,"rank":16,"preserve_w":10.0,"pres2_w":300.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":2.0,"gamma":0.5} fit[dW=0.0753 gap=0.4432] -> kl=0.01465 top1=0.9574
- 15:47:21Z c3t017 [fail] {"l_min":9,"l_max":23,"rank":16,"preserve_w":10.0,"pres2_w":300.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":1.0,"gamma":0.5} fit[dW=0.0688 gap=0.4458] -> kl=0.01614 top1=0.9556
- 15:49:36Z c3t018 [fail] {"l_min":13,"l_max":23,"rank":16,"preserve_w":10.0,"pres2_w":300.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":1.0,"gamma":0.5} fit[dW=0.072 gap=0.4283] -> kl=0.007648 top1=0.9695
- 15:52:06Z c3t019 [fail] {"l_min":11,"l_max":23,"rank":8,"preserve_w":10.0,"pres2_w":300.0,"overcorrect":0.0,"knn_k":5,"steps":200,"lr":0.0003,"think_w":1.0,"mean_w":1.0,"gamma":0.5} fit[dW=0.0611 gap=0.4916] -> kl=0.01156 top1=0.9629
- 15:55:22Z c3t020 [fail] {"l_min":11,"l_max":23,"rank":32,"preserve_w":10.0,"pres2_w":300.0,"overcorrect":0.0,"knn_k":5,"steps":300,"lr":0.0002,"think_w":1.0,"mean_w":1.0,"gamma":0.5} fit[dW=0.083 gap=0.3959] -> kl=0.01159 top1=0.9618
- 15:58:37Z c3t021 [fail] {"l_min":11,"l_max":23,"rank":16,"preserve_w":30.0,"pres2_w":300.0,"overcorrect":0.0,"knn_k":5,"steps":300,"lr":0.0002,"think_w":1.5,"mean_w":1.0,"gamma":0.7} fit[dW=0.0815 gap=0.5066] -> kl=0.01007 top1=0.9655
