#!/bin/bash
# W4: mixed-effort arms that give debugging tasks a learning signal (see effort_finish_probe.py).
#   LAB_PAID_MAX_USD=45 scripts/tinker/launch_w4.sh <arm> [<arm> ...]
# Arms are named here so every launch is one plain command and the settings stay on record.
HERE="$(cd "$(dirname "$0")" && pwd)"
LAB="$HOME/.inkling-abl/tinker_lab"
# W16 anchor loss segment: answer (thinking length stays free, W15) or all (W14). Override per launch.
ANCHOR_PART="${ANCHOR_PART:-answer}"
# W16 batch: 32 prompts = 10 debug + 7 maths + 7 creative + 3 chat + 3 roleplay + 2 qa. Every
# category is named and the rounded shares sum to exactly 32; pick_batch errors otherwise.
W16_FRAC="debug=0.3,maths=0.22,creative=0.22,chat=0.09,roleplay=0.09,qa=0.06"
COMMON=(--lr 5e-5 --steps 30 --group 8 --prompts-per-step 16 --rank 32 --loss importance_sampling
        --judge --save-every 10 --prompts prompts_train_code.jsonl --no-loop --no-overlong --no-length
        --efforts 0.5,0.6,0.7,0.8,0.9,0.9 --cat-frac debug=0.5)
BREV=(--brevity-cats debug,maths --brevity-gate 0.99)
# W7: same recipe with the length term on and Q&A/roleplay targets taken from the stock model's
# own answers (anchors.py), so shorter thinking does not drag answer length down with it.
LEN=(--lr 1e-4 --steps 30 --group 8 --prompts-per-step 16 --rank 32 --loss importance_sampling
     --judge --save-every 10 --prompts prompts_train_code_len.jsonl --no-loop --no-overlong --no-brevity
     --efforts 0.5,0.6,0.7,0.8,0.9,0.9 --cat-frac debug=0.5)
for ARM in "$@"; do
  case "$ARM" in
    w4_mix)       MODE=paid; X=(--no-brevity) ;;
    w4_mix_attn)  MODE=paid; X=(--no-brevity --attn-only) ;;
    w4_mix_brev)  MODE=paid; X=("${BREV[@]}") ;;
    w4_mix_lr1e4) MODE=paid; X=(--no-brevity --lr 1e-4) ;;
    w4_all_attn)  MODE=free; X=("${BREV[@]}" --attn-only --div) ;;
    # Tight budget: tests must pass within 2,048 thinking tokens (stock e0.9 needs ~11K on
    # these tasks when it finishes at all), so lower efforts are included to seed the signal.
    w4_cap2k)     MODE=free; X=(--no-brevity --think-cap 2048 --efforts 0.3,0.4,0.5,0.6,0.7,0.9) ;;
    # W5: attention-only learned too slowly at 5e-5 (debug reward 0.4 at step 10 against 0.9+
    # for attention+experts at 1e-4), so test whether a higher LR closes the gap. Attention-only
    # is the form that ships as a drop-in (experts stay byte-identical).
    w5_attn_lr1e4) MODE=paid; X=(--no-brevity --attn-only --lr 1e-4) ;;
    w5_attn_lr2e4) MODE=free; X=(--no-brevity --attn-only --lr 2e-4) ;;
    w5_attn_lr4e4) MODE=free; X=(--no-brevity --attn-only --lr 4e-4) ;;
    # W6: second seed of the leading recipe, to check the headline result reproduces.
    w6_lr1e4_s1)   MODE=free; X=(--no-brevity --lr 1e-4 --seed 1) ;;
    w7_attn_len)   MODE=free; COMMON=("${LEN[@]}"); X=(--attn-only) ;;
    w7_full_len)   MODE=paid; COMMON=("${LEN[@]}"); X=(--lr 5e-5 --steps 20) ;;
    # W8: attention-only weak spots. More training fixed temperature-1.0 writing but cost hard
    # maths at effort 0.3 (never trained: AIME e0.3 90% stock -> 87 / 81 / 76 as training grew).
    # e03 adds effort 0.3 to the mix; r128 tests whether capacity closes the gap to full LoRA.
    w8_attn_len_e03) MODE=free; COMMON=("${LEN[@]}"); X=(--attn-only --efforts 0.3,0.5,0.6,0.7,0.8,0.9,0.9) ;;
    w8_attn_r128)    MODE=free; COMMON=("${LEN[@]}"); X=(--attn-only --rank 128 --efforts 0.3,0.5,0.6,0.7,0.8,0.9,0.9) ;;
    # W9: owner dropped the attention-only requirement for v3 (2026-11-25), so attention-only
    # arms were stopped. These refine the full recipe (w4_mix: attention+experts, 5e-5, 20 steps).
    w9_full_s1)    MODE=free; X=(--no-brevity --steps 20 --seed 1) ;;
    w9_full_div)   MODE=free; X=(--no-brevity --steps 20 --div) ;;
    # W10: how many steps does the full recipe need? Seeds differ at 5e-5: w4_mix was done by
    # step 16, w9_full_div by 19, w9_full_s1 still stalled at effort 0.9 at step 20.
    # s1b continues s1 for 15 more steps; lr75 tries 7.5e-5. Both save every 5 steps.
    w10_s1b)       MODE=free; X=(--no-brevity --steps 15 --seed 11 --save-every 5 --resume-state "$S1_STATE") ;;
    w10_lr75)      MODE=free; X=(--no-brevity --steps 30 --lr 7.5e-5 --save-every 5) ;;
    # W11: v3 trains with thinking-off prompts in the mix; check that a quarter of groups with
    # thinking off does not disturb the stall fix (30 steps, checkpoint every 5).
    w11_off)       MODE=free; X=(--no-brevity --steps 30 --save-every 5 --off-frac 0.25) ;;
    # W12: training past the fix shortens Q&A answers again even at 5e-5 (step 35: 199 tokens
    # against stock's 312, -0.041 +- 0.021 on the held-out grader). Does anchoring Q&A/roleplay
    # length to stock's own answers hold it through 35 steps?
    w12_full_len)  MODE=free; COMMON=("${LEN[@]}"); X=(--lr 5e-5 --steps 35 --save-every 5) ;;
    # W13: rehearsal of the GPU run's batch shape (32 prompts x 8, 8 debug prompts per step as
    # before, 25% thinking-off). Does a wider, more varied batch fix the stall with less Q&A drift?
    w13_wide)      MODE=free; X=(--no-brevity --steps 35 --save-every 5 --off-frac 0.25 --prompts-per-step 32 --cat-frac debug=0.25) ;;
    # Same with Q&A prompts left out of training. If Q&A still shortens, the drift is spillover
    # from the other tasks; if it does not, the training judge's taste for short answers is the cause.
    w13_wide_noqa) MODE=free; X=(--no-brevity --steps 35 --save-every 5 --off-frac 0.25 --prompts-per-step 32 --cat-frac debug=0.25,qa=0) ;;
    # U1: first uncensor arm. w13_wide's recipe plus the v3rl reward (quality - 0.5*held_back) on
    # harmful/chat rows and one system prompt per group from systems_spread.jsonl (none -> strongly
    # permissive), so groups the stock model refuses at every rollout still meet conditions with signal.
    # Run uncensor_probe.py first. Judge spend also counts against LAB_JUDGE_MAX_USD (run_arm_live: 6).
    u1_probe_train) MODE=free; X=(--no-brevity --steps 35 --save-every 5 --off-frac 0.25 --prompts-per-step 32
                                  --uncensor --systems systems_spread.jsonl --prompts "$HOME/.inkling-abl/v3_private/pool_v3/pool.jsonl"
                                  --cat-frac debug=0.25,harmful=0.31) ;;
    # U1: first uncensoring run from stock. The signal test (uncensor_probe.py, tag stock) showed system
    # prompts do not help (all-refused groups 60-69% at every level), thinking-off gives the most mixed
    # groups (36%), and crime, hacking and sexual prompts are refused at nearly every rollout. So: no
    # system-prompt spread, the uncensor reward on every judged category, 12 harmful groups per step.
    u1_stock)      MODE=paid; X=(--no-brevity --steps 40 --save-every 5 --off-frac 0.25 --prompts-per-step 32
                                  --uncensor --uncensor-cats harmful,chat,creative,roleplay
                                  --prompts "$HOME/.inkling-abl/v3_private/pool_v3/pool.jsonl"
                                  --cat-frac debug=0.25,harmful=0.375) ;;
    # W14: self-distillation anchor. w11_off's recipe for 40 steps plus cross-entropy on 16 of
    # stock's own best non-debug answers (build_anchors.py) in every optimizer step, to pin Q&A
    # length/quality while RL fixes debugging. w14_ctrl is the same seed without anchors.
    # Both use the new judge limiter/retry/drop defaults (no silent heuristic fallback).
    w14_anchor)    MODE=free; X=(--no-brevity --steps 40 --save-every 5 --off-frac 0.25
                                 --anchors "$LAB/anchors/stock_best.jsonl" --anchor-n 16 --anchor-weight 1.0) ;;
    # W15: W14's anchor held back the stall fix (87% solved at step 25, control 100% at 15) because
    # it also pinned stock's thinking. Anchor the answer only; keep trainer state from step 20.
    w15_anchor_ans) MODE=free; X=(--no-brevity --steps 40 --save-every 5 --off-frac 0.25 --state-from 20
                                  --anchors "$LAB/anchors/stock_best.jsonl" --anchor-n 16 --anchor-weight 1.0
                                  --anchor-part answer) ;;
    # W14b: continue w14_anchor from step 40 for 20 more steps. Its no-code rate was still falling
    # (85/79/67/48/21% at steps 20-40) with Q&A held at stock, so see whether it reaches 0.
    w14b_anchor)   MODE=free; X=(--no-brevity --steps 20 --seed 21 --save-every 5 --off-frac 0.25 --state-from 5
                                 --resume-state "tinker://24bfeac4-e1cb-509f-980b-824fbc07c0c1:train:0/weights/w14_anchor-step40"
                                 --anchors "$LAB/anchors/stock_best.jsonl" --anchor-n 16 --anchor-weight 1.0) ;;
    # W17: the final candidates. Continue w15_anchor_ans from step 40 (answer-only anchor, most of the
    # stall fix in place) on the wide file (1,185 prompts) for 25 steps, state kept at every save.
    #   w17_qc    anchor only qa+chat (the categories that shorten); creative/roleplay left to RL,
    #             since the all-category anchor halved the creative gain (+0.04 vs +0.08)
    #   w17_qc03  the same at anchor weight 0.3
    #   w17_all   all non-debug categories anchored (w15's recipe on the wide file): the reference
    w17_qc|w17_qc03|w17_all)
        MODE=free
        case "$ARM" in
          w17_qc)   AX=(--anchor-cats qa,chat --anchor-weight 1.0) ;;
          w17_qc03) AX=(--anchor-cats qa,chat --anchor-weight 0.3) ;;
          w17_all)  AX=(--anchor-weight 1.0) ;;
        esac
        X=(--no-brevity --steps 25 --save-every 5 --state-from 15 --off-frac 0.25 --seed 17
           --resume-state "tinker://827d8b0c-829c-5fd3-8ed2-883845899a45:train:0/weights/w15_anchor_ans-step40"
           --prompts "$LAB/prompts_train_wide.jsonl" --prompts-per-step 32
           --cat-frac "debug=0.31,maths=0.19,creative=0.16,qa=0.16,roleplay=0.09,chat=0.09"
           --anchors "$LAB/anchors/wide_all.jsonl" --anchor-n 16 --anchor-part answer "${AX[@]}") ;;
    # W18: the full run from the stock model on the wide file, no warm start: 70 steps, anchor on
    # qa+chat answers only. Slower than W17 (about 5 h) but every step sees the wide data.
    w18_full)      MODE=free; X=(--no-brevity --steps 70 --save-every 10 --state-from 40 --off-frac 0.25 --seed 18
                                 --prompts "$LAB/prompts_train_wide.jsonl" --prompts-per-step 32
                                 --cat-frac "debug=0.31,maths=0.19,creative=0.16,qa=0.16,roleplay=0.09,chat=0.09"
                                 --anchors "$LAB/anchors/wide_all.jsonl" --anchor-n 16 --anchor-part answer
                                 --anchor-weight 1.0) ;;
    # W19: Prime continuation. w18_full's recipe continued from its step-60 / step-70 state on the
    # context file (build_wide_train.py --allow-context: + system/multi-turn roleplay and chat, + long
    # answers), with more roleplay per step, to close the small roleplay dip seen after merging.
    # Anchors: wide_all + ctx_new (stock's best answers for the new rows) = anchors/ctx_all.jsonl.
    w19_s60|w19_s70)
        MODE=free
        X=(--no-brevity --steps 25 --save-every 5 --state-from 15 --off-frac 0.25 --seed 19
           --resume-state "tinker://deab96f0-da13-5ac5-a320-3226376dd1cd:train:0/weights/w18_full-step${ARM#w19_s}"
           --prompts "$LAB/prompts_train_wide_ctx.jsonl" --prompts-per-step 32
           --cat-frac "debug=0.25,maths=0.125,creative=0.1875,qa=0.15625,roleplay=0.1875,chat=0.09375"
           --anchors "$LAB/anchors/ctx_all.jsonl" --anchor-n 16 --anchor-part answer --anchor-weight 1.0) ;;
    w14_ctrl)      MODE=free; X=(--no-brevity --steps 40 --save-every 5 --off-frac 0.25) ;;
    # W16: final run on the wide file (build_wide_train.py: lab file + harmless pool rows the plain
    # path renders faithfully). w15_anchor_ans recipe, 32 prompts per step, 80 steps, two seeds.
    # Build anchors first: build_anchors.py --prompts prompts_train_wide.jsonl --out anchors/wide_best.jsonl
    w16_wide_s0)   MODE=free; X=(--no-brevity --steps 80 --save-every 5 --off-frac 0.25 --state-from 30
                                 --prompts "$LAB/prompts_train_wide.jsonl" --prompts-per-step 32 --cat-frac "$W16_FRAC"
                                 --anchors "$LAB/anchors/wide_best.jsonl" --anchor-n 24 --anchor-weight 1.0
                                 --anchor-part "$ANCHOR_PART" --seed 0) ;;
    w16_wide_s1)   MODE=free; X=(--no-brevity --steps 80 --save-every 5 --off-frac 0.25 --state-from 30
                                 --prompts "$LAB/prompts_train_wide.jsonl" --prompts-per-step 32 --cat-frac "$W16_FRAC"
                                 --anchors "$LAB/anchors/wide_best.jsonl" --anchor-n 24 --anchor-weight 1.0
                                 --anchor-part "$ANCHOR_PART" --seed 1) ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
  esac
  nohup "$HERE/run_arm_live.sh" "$ARM" "$MODE" "${COMMON[@]}" "${X[@]}" > "$LAB/logs/arm_$ARM.out" 2>&1 &
  echo "$(date '+%H:%M') launched $ARM ($MODE)" | tee -a "$LAB/status.txt"
done
