"""Optuna ask/tell loop for MPOA families (c1, c2) over the shared queue.

  search.py <family c1|c2> <n_trials> <inflight> [--stop HH:MM]   (UTC stop time; default 14:00)

Per trial: ask -> build compact candidate -> enqueue "screen" -> wait for screen + KL rows ->
  if screen refusal <= 5% and KL <= 0.04: enqueue "full" and wait -> tell.
Objectives (minimise): val refusal (full: mean of off and e0.9; screen-only: screen off rate), KL (HF, kl_worker).
Constraints (<= 0 feasible): MMLU-300 >= 75 (orig 77 - 2), GSM8K-100 >= 92 (orig 97 - 5).
Resumable: RUNNING trials keep their cand_id/stage in user attrs and are picked up again on restart.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import optuna
from optuna.trial import TrialState

sys.path.insert(0, os.path.dirname(__file__))
import enqueue as Q
import metrics as MET
import mpoa_lib as M

R = M.R
FAM = sys.argv[1]
N_TRIALS = int(sys.argv[2])
INFLIGHT = int(sys.argv[3])
STOP = sys.argv[sys.argv.index("--stop") + 1] if "--stop" in sys.argv else "14:00"
MMLU_MIN, GSM_MIN = 74.0, 92.0   # TARGETS.md: orig 77 - 3, orig 97 - 5 (update when fleet re-scores orig)
LAM_LO = 0.3 if FAM[-1] in "bc" else 1e-2   # strong ridge doubled KL in round 1
SCREEN_REF_MAX, SCREEN_KL_MAX = (0.01, 0.04) if FAM[-1] in "cd" else (0.05, 0.06)   # 15:10Z: promote c1c/c1d screens <= 1%
TIMEOUT_S = 50 * 60
LOG = f"{R}/methods/mpoa/LOG.md"
ORPHAN_FULL = f"{R}/state/optuna/{FAM}_orphans_in_full.txt"
STATUS = f"{R}/state/agents/mpoa.txt" if FAM == "c1" else f"{R}/state/agents/mpoa_{FAM}.txt"
os.makedirs(f"{R}/state/optuna", exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def past_stop() -> bool:
    h, m = map(int, STOP.split(":"))
    t = datetime.now(timezone.utc)
    return t.hour * 60 + t.minute >= h * 60 + m


def constraints(ft) -> list[float]:
    return ft.user_attrs.get("constraints", [0.0] * 6)


# ---------- search spaces ----------

def kernel_space(t) -> dict:
    if FAM.endswith("b"):
        # strong space (lead 14:18Z): reflection allowed, all of L4-23, no late-layer prior, MP pre
        return {
            "r": t.suggest_categorical("r", [2, 1]),
            "conv": t.suggest_categorical("conv", ["svd1", "plain", "svd2", "exact4"]),
            "conv_scope": "all", "mp": "pre", "l_min": t.suggest_int("l_min", 4, 11),
            "mu": t.suggest_float("mu", 4.0, 23.0),
            "width": t.suggest_float("width", 2.0, 14.0),
            "peak": t.suggest_float("peak", 1.0, 4.0),
            "floor": t.suggest_float("floor", 0.5, 4.0),
            "loop_m": t.suggest_float("loop_m", 0.4, 1.5),
            "late_m": t.suggest_float("late_m", 0.5, 1.5),
        }
    return {
        "r": t.suggest_categorical("r", [1, 2]),
        "conv": t.suggest_categorical("conv", ["exact4", "svd2", "svd1", "plain"]),
        "conv_scope": t.suggest_categorical("conv_scope", ["all", "leaky"]),
        "mp": t.suggest_categorical("mp", ["pre", "none"]),
        "mu": t.suggest_float("mu", 4.0, 23.0),
        "width": t.suggest_float("width", 1.5, 12.0),
        "peak": t.suggest_float("peak", 0.3, 2.5),
        "floor": t.suggest_float("floor", 0.0, 1.2),
        "loop_m": t.suggest_float("loop_m", 0.3, 1.5),
        "late_m": t.suggest_float("late_m", 0.0, 1.2),
        "l_min": t.suggest_int("l_min", 4, 12),
    }


def split_space(t) -> dict:
    """Hygiene study (lead 14:45Z): separate strengths before/inside/after the loop; loop kept light."""
    d = FAM.endswith("d")   # c1d (15:10Z): loop strength ~2 is required for refusal; cut KL outside the loop
    s_pre = t.suggest_float("s_pre", 0.0, 1.5 if d else 4.5)
    s_loop = t.suggest_float("s_loop", 1.5 if d else 0.0, 2.7 if d else 1.5)
    s_post = t.suggest_float("s_post", 0.0 if d else 1.0, 2.5 if d else 4.5)
    return {"r": t.suggest_categorical("r", [1, 2]),
            "conv": t.suggest_categorical("conv", ["svd1", "plain", "svd2"]),
            "conv_scope": "all", "mp": "pre", "s_pre": s_pre, "s_loop": s_loop, "s_post": s_post,
            "s_list": [s_pre] * 2 + [s_loop] * 12 + [s_post] * 6}


def params_for(t) -> dict:
    p = split_space(t) if FAM[-1] in "cd" else kernel_space(t)
    p["harmless_proj"] = True
    if p["conv_scope"] == "leaky":
        p["exact_layers"] = LEAKY
    if FAM.startswith("c2"):
        la = t.suggest_float("lam_atoms", LAM_LO, 1e2, log=True)
        p["protect"] = {"harmfulness": t.suggest_float("lam_harm", LAM_LO, 1e2, log=True),
                        "math": la, "code": la, "honest": la, "crisis": la}
        p["protect_k"] = t.suggest_int("protect_k", 1, 2)
        p["think_w"] = t.suggest_float("think_w", 0.0, 1.0)
    return p


# Calibration seeds: fixed trials built and scored on the fleet like any other trial (constant s = kernel floor = peak).
BASE = {"mu": 13.0, "width": 12.0, "peak": 1.0, "floor": 1.0, "loop_m": 1.0, "late_m": 1.0, "l_min": 4,
        "r": 2, "conv": "exact4", "conv_scope": "all", "mp": "pre"}
SEEDS = {"c1": [
    {},                                         # exact4, s=1, L4-23, MP
    {"conv": "plain"},                          # C0-like projection + MP: value of the leak fix
    {"conv": "svd1"},                           # rank-r partial fix (same rank as plain, leak 0.03-0.21)
    {"conv": "svd2"},                           # rank-2r partial fix (leak <= 0.07)
    {"conv_scope": "leaky"},                    # exact4 on L4-8, 17 only; plain elsewhere
    {"mp": "none"},                             # exact4 without norm preservation
    {"l_min": 11},                              # exact4 on L11-23 only
    {"peak": 0.7, "floor": 0.7},                # exact4, s=0.7
]}
C2B = {"lam_atoms": 1.0, "lam_harm": 1.0, "protect_k": 2, "think_w": 0.0}
SEEDS["c2"] = [
    {"conv": "svd2"},                                         # ridge lam=1 on both groups
    {"conv": "svd2", "lam_harm": 0.1},                        # strong harmfulness orthogonalisation
    {"conv": "svd2", "lam_atoms": 0.1},                       # strong atom orthogonalisation
    {"conv": "svd2", "lam_harm": 10.0, "lam_atoms": 0.1},     # atoms only
    {"conv": "svd2", "think_w": 0.5},                         # + thinking-position direction
    {"conv": "exact4"},                                       # exact fix with ridge lam=1
    {"conv": "svd1", "peak": 1.3, "floor": 1.3},              # partial fix, mild reflection
    {"conv": "svd2", "lam_harm": 0.3, "lam_atoms": 0.3, "think_w": 0.3},
]
for ov in SEEDS["c2"]:
    for k, v in C2B.items():
        ov.setdefault(k, v)
for fam in SEEDS:
    for ov in SEEDS[fam]:
        for k, v in BASE.items():
            ov.setdefault(k, v)
# Hygiene seeds: strength outside the loop, light inside it (c1t013 had s~2.0-2.4 on looped L7-17 -> 25% e0.9 repetition).
SEEDS["c1c"] = [
    {"r": 1, "conv": "svd1", "s_pre": 3.0, "s_loop": 0.5, "s_post": 3.0},
    {"r": 1, "conv": "svd1", "s_pre": 3.0, "s_loop": 1.0, "s_post": 3.0},
    {"r": 1, "conv": "svd1", "s_pre": 4.0, "s_loop": 0.0, "s_post": 4.0},
    {"r": 2, "conv": "svd1", "s_pre": 3.0, "s_loop": 0.7, "s_post": 3.0},
    {"r": 1, "conv": "plain", "s_pre": 4.0, "s_loop": 1.0, "s_post": 4.0},
    {"r": 2, "conv": "svd1", "s_pre": 2.0, "s_loop": 1.0, "s_post": 4.0},
]
# c1d seeds: c1t013/A-like loop strength (~2), light or no edit outside the loop to cut KL.
SEEDS["c1d"] = [
    {"r": 2, "conv": "svd1", "s_pre": 0.0, "s_loop": 2.0, "s_post": 1.0},
    {"r": 1, "conv": "svd1", "s_pre": 0.0, "s_loop": 2.2, "s_post": 1.0},
    {"r": 2, "conv": "svd1", "s_pre": 0.0, "s_loop": 2.2, "s_post": 0.5},
    {"r": 1, "conv": "plain", "s_pre": 0.0, "s_loop": 2.4, "s_post": 0.8},
    {"r": 2, "conv": "svd1", "s_pre": 1.0, "s_loop": 1.8, "s_post": 1.0},
    {"r": 1, "conv": "svd1", "s_pre": 0.0, "s_loop": 2.6, "s_post": 0.0},
    {"r": 2, "conv": "svd1", "s_pre": 0.0, "s_loop": 2.0, "s_post": 0.0},
    {"r": 2, "conv": "plain", "s_pre": 0.0, "s_loop": 2.0, "s_post": 1.0},
]
LEAKY = [4, 5, 6, 7, 8, 17]


# ---------- bookkeeping ----------

def log_line(cid: str, p: dict, m: dict, verdict: str) -> None:
    keys = ["kl", "top1", "screen_refusal_off", "screen_mmlu300", "screen_harmful_off_rep", "refusal_off", "refusal_e09",
            "mmlu300", "gsm8k100", "harmful_e09_cap_hit", "harmful_e09_rep", "judge_nonevasive_e09"]
    short = {k: v for k, v in p.items() if k not in ("harmless_proj",)}
    vals = " ".join(f"{k}={m[k]:.4g}" if isinstance(m.get(k), (int, float)) else "" for k in keys if k in m)
    with open(LOG, "a") as f:
        f.write(f"- {now()} {cid} [{verdict}] {json.dumps(short, separators=(',', ':'))} -> {vals}\n")


def status(msg: str) -> None:
    with open(STATUS, "w") as f:
        f.write(f"{now()} mpoa[{FAM}]: {msg}\n")


def start_trial(study) -> dict | None:
    t = study.ask()
    p = params_for(t)
    cid = f"{FAM}t{t.number:03d}"
    M.write_candidate(cid, FAM[:2], p)
    Q.enqueue([cid], "screen", 9.0 if FAM.endswith("d") else (8.0 if FAM.endswith("c") else 5.5), FAM[:2])
    t.set_user_attr("cand_id", cid)
    t.set_user_attr("stage", "screen")
    t.set_user_attr("params_json", json.dumps(p, default=str))
    return {"trial": t, "cid": cid, "p": p, "t0": time.time()}


def finish(study, job: dict, m: dict, verdict: str) -> None:
    t = job["trial"]
    if not job.get("orphan"):
        for k, v in m.items():
            if isinstance(v, (int, float, str)):
                t.set_user_attr(f"m_{k}", v)
    if verdict == "fail":
        if not job.get("orphan"):
            study.tell(t, state=TrialState.FAIL)
    else:
        if verdict == "full":
            ref = (m["refusal_off"] + m["refusal_e09"]) / 2
            # TARGETS.md hard constraints; a metric the full stage did not report counts as satisfied (logged).
            c = [MMLU_MIN - m["mmlu300"], GSM_MIN - m["gsm8k100"],
                 0.0, m.get("harmful_e09_cap_hit", 0.0) - 0.196,   # owner 15:05Z: rep rule dropped; cap <= OU (19.6%)
                 1.0 - min(m.get("probe_honest_limit_off", 1.0), m.get("probe_honest_limit_e09", 1.0)),
                 m.get("overrefusal200_off", 0.0) - 0.02]
        else:
            ref = m["screen_refusal_off"]
            c = [MMLU_MIN - m["screen_mmlu300"], 0.0, 0.0, 0.0, 0.0, 0.0]
        if job.get("orphan"):
            # trial was marked FAIL by the old 50-min timeout while its candidate stayed queued: record it as a new trial
            ft = job["frozen"]
            ua = {k: v for k, v in m.items() if isinstance(v, (int, float, str))}
            ua.update({"cand_id": job["cid"], "constraints": c, "recovered_from": ft.number})
            study.add_trial(optuna.trial.create_trial(params=ft.params, distributions=ft.distributions,
                                                      values=[ref, m["kl"]], user_attrs=ua))
        else:
            t.set_user_attr("constraints", c)
            study.tell(t, [ref, m["kl"]])
    log_line(job["cid"], job["p"], m, verdict)


def step(study, job: dict, allm: dict) -> bool:
    """Advance one in-flight job. Returns True when it is finished."""
    t, cid = job["trial"], job["cid"]
    m = allm.get(cid, {})
    if any(k.startswith("_error") for k in m):   # no wall-clock timeout: queue waits are expected
        finish(study, job, m, "fail")
        return True
    stage = job["stage"]
    if stage == "screen":
        if "kl" in m and "screen_refusal_off" in m and "screen_mmlu300" in m:
            if m["screen_refusal_off"] <= SCREEN_REF_MAX and m["kl"] <= SCREEN_KL_MAX:
                Q.enqueue([cid], "full", 9.7 if FAM[-1] in "cd" else 7.0, FAM[:2])
                job["stage"] = "full"
                job["t0"] = time.time()
                if not job.get("orphan"):
                    t.set_user_attr("stage", "full")
                else:
                    open(ORPHAN_FULL, "a").write(cid + "\n")
            else:
                finish(study, job, m, "screen")
                return True
    elif stage == "full":
        if all(k in m for k in ("refusal_off", "refusal_e09", "mmlu300", "gsm8k100")):
            finish(study, job, m, "full")
            return True
    return False


def main() -> None:
    sampler = optuna.samplers.TPESampler(constraints_func=constraints, n_startup_trials=8, multivariate=True,
                                         group=True, seed=17 if FAM == "c1" else 29)
    study = optuna.create_study(study_name=FAM, storage=f"sqlite:///{R}/state/optuna/{FAM}.db",
                                directions=["minimize", "minimize"], sampler=sampler, load_if_exists=True)
    jobs: list[dict] = []
    for ft in study.get_trials(states=(TrialState.RUNNING,)):
        cid = ft.user_attrs.get("cand_id")
        if not cid:
            study.tell(ft.number, state=TrialState.FAIL)
            continue
        tr = optuna.trial.Trial(study, ft._trial_id)
        p = json.load(open(f"{R}/methods/mpoa/cands/{cid}/spec.json"))["params"]
        jobs.append({"trial": tr, "cid": cid, "p": p, "stage": ft.user_attrs.get("stage", "screen"),
                     "t0": time.time()})
    recovered = {ft.user_attrs.get("cand_id") for ft in study.get_trials(states=(TrialState.COMPLETE,))}
    in_full = set(open(ORPHAN_FULL).read().split()) if os.path.exists(ORPHAN_FULL) else set()
    for ft in study.get_trials(states=(TrialState.FAIL,)):
        cid = ft.user_attrs.get("cand_id")
        if not cid or cid in recovered:
            continue
        p = json.load(open(f"{R}/methods/mpoa/cands/{cid}/spec.json"))["params"]
        jobs.append({"trial": None, "frozen": ft, "orphan": True, "cid": cid, "p": p,
                     "stage": "full" if cid in in_full else "screen", "t0": time.time()})
    if not study.trials:
        for ov in SEEDS.get(FAM, []):
            study.enqueue_trial(ov)
    started = len(study.trials) - len(study.get_trials(states=(TrialState.WAITING,)))
    while True:
        allm = MET.collect([f"{FAM}t"], exclude_pods=("ink-eval-2",))   # Pareto uses fleet rows only
        jobs = [j for j in jobs if not step(study, j, allm)]
        while len(jobs) < INFLIGHT and started < N_TRIALS and not past_stop():
            try:
                j = start_trial(study)
            except Exception:
                traceback.print_exc()
                time.sleep(30)
                break
            j["stage"] = "screen"
            jobs.append(j)
            started += 1
        done = len(study.get_trials(states=(TrialState.COMPLETE, TrialState.FAIL)))
        status(f"{done} done, {len(jobs)} in flight ({','.join(j['cid'] + ':' + j['stage'] for j in jobs)}); "
               f"started {started}/{N_TRIALS}; stop {STOP}Z")
        if not jobs and (started >= N_TRIALS or past_stop()):
            break
        time.sleep(20)
    print("search finished", FAM)


if __name__ == "__main__":
    main()
