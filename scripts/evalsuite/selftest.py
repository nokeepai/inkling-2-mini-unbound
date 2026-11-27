"""End-to-end self-test against the local mock server: no GPU, no network, no judge spend.

  python3 selftest.py [--keep]

Checks: ported metric functions; the template port against the official jinja template (if jinja2 and
the template file are present); channel parsing across chunk boundaries; sysrules rule names; then all
seven suites x 2 repeats against a vLLM-style mock (reasoning field, mock judges), the main suite against
a llama.cpp-style mock with raw channel markers, the no-reasoning-parser warning, --judge-only, compare.py.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mathcheck as C  # noqa: E402
import metrics as M  # noqa: E402
import pub as P  # noqa: E402
import stats as ST  # noqa: E402
import sysrules as SR  # noqa: E402
import template as tpl  # noqa: E402
from client import ChannelParser  # noqa: E402

DATA = Path("~/.inkling-abl/tinker_lab").expanduser()
AGENT = HERE.parent / "tinker" / "agent_hard_tasks"
U_TOOLS = Path(os.environ.get("EVALSUITE_UBENCH_TOOLS", "~/Code/uncensoredbench/tools")).expanduser()
PY = sys.executable
RESULTS: list[str] = []


def check(cond: bool, label: str) -> None:
    RESULTS.append(("PASS " if cond else "FAIL ") + label)
    print(RESULTS[-1], flush=True)


def official_template() -> Path | None:
    for base in [HERE, *HERE.parents, Path("~/Code/inkling-abl").expanduser()]:
        p = base / "out" / "final" / "configs" / "chat_template.jinja"
        if p.exists():
            return p
    return None


def unit_checks() -> None:
    M.checks()
    check(True, "metrics: lab inline checks")
    P.checks()
    check(True, "pub: MMLU answer-line extractor cases, MBPP and HumanEvalFix program builders")
    C.checks()
    check(True, "mathcheck: number reading (units, currency, separators, percent, fractions, negatives, "
                "wrong-with-units, two numbers) and MATH is_equiv")
    ST.checks()
    check(True, "stats: Wilson, exact McNemar and paired bootstrap on hand-computed cases")
    p = ChannelParser(raw_markers=True)
    out = []
    for chunk in ["<|message_model|><|content_", "thinking|>Hmm ", "ok<|end_", "message|><|message_model|>",
                  "<|content_text|>if a < b: x", "<|end_message|>"]:
        out += p.feed(chunk)
    think = "".join(t for c, t in out if c == "think")
    ans = "".join(t for c, t in out if c == "answer")
    check(think == "Hmm ok" and ans == "if a < b: x" and p.think_closed, f"parser: split markers ({think!r}|{ans!r})")
    p2 = ChannelParser(raw_markers=False)
    got = p2.feed("<think>plan</think>Answer <b>bold</b>")
    check([c for c, _ in got if c == "think"] and "".join(t for c, t in got if c == "answer") == "Answer <b>bold</b>",
          "parser: <think> tags, html in answer kept")
    tasks = [json.loads(x) for x in (DATA / "sysrules_tasks.jsonl").read_text().splitlines() if x.strip()]
    names = {r[0] for r in SR.rule_pool("X", tasks[0]["params"])}
    check(all(set(t["rule_names"]) <= names for t in tasks), "sysrules: every task's rules resolve in the port")
    template_check()


def template_check() -> None:
    path = official_template()
    try:
        import jinja2
    except ImportError:
        jinja2 = None  # type: ignore[assignment]
    if path is None or jinja2 is None:
        check(True, "template: official jinja not available here, skipped")
        return
    src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", path.read_text())

    def raise_exception(msg: str) -> None:
        raise ValueError(msg)
    t = jinja2.Environment().from_string(src)
    convs = [[{"role": "user", "content": "hi"}],
             [{"role": "system", "content": "S1"}, {"role": "user", "content": "q"},
              {"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}],
             [{"role": "system", "content": "only system"}]]
    ok = all(t.render(messages=c, reasoning_effort=e, raise_exception=raise_exception) == tpl.render(c, e)
             for c in convs for e in (0, 0.3, 0.9, None, "high"))
    check(ok, f"template: port matches {path.name} (3 conversations x 5 efforts)")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_mock(style: str) -> tuple[subprocess.Popen[bytes], int]:
    port = free_port()
    proc = subprocess.Popen([PY, str(HERE / "mock_server.py"), "--port", str(port), "--style", style,
                             "--data-dir", str(DATA), "--agent-dir", str(AGENT)])
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return proc, port
        except OSError:
            time.sleep(0.1)
    raise SystemExit("mock did not start")


def run_suite(port: int, tag: str, out: Path, extra: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [PY, str(HERE / "run_suite.py"), "--base-url", f"http://127.0.0.1:{port}/v1", "--model", "mock-inkling",
           "--tag", tag, "--out-root", str(out), "--data-dir", str(DATA), "--agent-dir", str(AGENT), *extra]
    env = {**os.environ, "EVALSUITE_JUDGE_MOCK": "1", "OPENROUTER_API_KEY": "", "EVALSUITE_SPEND_DIR": str(out)}
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    print(f"  run_suite {tag}: rc={res.returncode} in {time.time() - t0:.0f}s", flush=True)
    if res.returncode != 0:
        print(res.stdout[-3000:], res.stderr[-3000:])
    return res


def load(out: Path, tag: str, suite: str, rep: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = out / tag / suite
    recs = [json.loads(x) for x in (d / f"r{rep}.jsonl").read_text().splitlines() if x.strip()]
    return recs, json.loads((d / f"r{rep}.summary.json").read_text())


def check_main(out: Path, tag: str) -> None:
    recs, s = load(out, tag, "main")
    cap = [r for r in recs if r["think_cap_hit"]]
    run = [r for r in recs if r["answer_runaway"]]
    check(len(cap) == 3 and all(r["think_tokens"] == 4096 and r["forced"] and r["answer"] for r in cap),
          f"{tag} main: 3 cap hits at 4096 thinking tokens, forced answers present ({len(cap)})")
    check(len(run) >= 6 and all(r["answer_tokens"] == 8192 for r in run),
          f"{tag} main: runaways cut at 8192 answer tokens ({len(run)})")
    check(all(r["loop_think"] for r in cap) and all(r["loop_answer"] for r in run),
          f"{tag} main: thinking loops on cap hits, answer loops on runaways")
    check(not any(r["loop_any"] for r in recs if not (r["think_cap_hit"] or r["answer_runaway"])),
          f"{tag} main: no false loop flags on clean rows")
    acc = s["by_category"]["maths"]["maths_acc"]
    check(0.4 <= acc <= 0.9 and s["n_failed"] == 0, f"{tag} main: maths accuracy {acc:.2f}, no failed requests")
    cr = s["by_category"]["creative"]
    check(0.5 <= cr["length_within_30pct"] < 1.0 and cr["top_names"], f"{tag} main: creative length "
          f"{cr['length_within_30pct']:.2f}, names {cr['top_names'][:2]}")


def check_field_run(out: Path, port: int) -> None:
    tag = "mock_field"
    check_main(out, tag)
    _, s = load(out, tag, "main")
    check(s["all"].get("judge_good_rate") is not None and s["all"].get("alt_good_rate") is not None,
          "field main: Bison and Haiku (mock) good-rates present")
    _, h = load(out, tag, "hardmath")
    check(0.5 <= h["all"]["maths_acc"] <= 0.85, f"field hardmath: accuracy {h['all']['maths_acc']:.2f}")
    _, c = load(out, tag, "code")
    check(abs(c["pass_at_1"] - 1 / 40) < 1e-9, f"field code: pass@1 {c['pass_at_1']:.3f} (1 correct of 40)")
    ar, a = load(out, tag, "agent_hard")
    t05 = next(r for r in ar if r["name"].startswith("t05"))
    check(abs(a["solve_rate"] - 14 / 15) < 1e-9 and a["repeat_edit_rate"] > 0 and a["no_code_rate"] > 0,
          f"field agent_hard: solve {a['solve_rate']:.2f}, repeats {a['repeat_edit_rate']:.2f}, "
          f"no-code {a['no_code_rate']:.2f}")
    check(t05["solved_at"] == 1 and t05["turns"][0]["cap_hit"] and t05["turns"][0]["forced"],
          "field agent_hard: t05 cap hit on turn 1, forced answer solves it")
    nr, n = load(out, tag, "agent_hard_noforce")
    t05n = next(r for r in nr if r["name"].startswith("t05"))
    check(t05n["solved_at"] == 2 and t05n["turns"][0]["no_code"] and not t05n["turns"][0]["forced"],
          "field agent_hard_noforce: t05 cap hit leaves no answer, solved on turn 2")
    _, rp = load(out, tag, "rp")
    check(rp["cross_turn_repeat"] > 0 and rp["haiku_good"] is not None and rp["bison_good"] is not None,
          f"field rp: cross-turn repeats {rp['cross_turn_repeat']:.2f}, judges present")
    _, sr = load(out, tag, "sysrules")
    check(sr["rule_compliance"] > 0.5 and sr["judged_rules_scored"],
          f"field sysrules: compliance {sr['rule_compliance']:.2f}, judge rules scored")
    agg = json.loads((out / tag / "main" / "summary.json").read_text())["agg"]
    check(agg["all.loop_any"]["n"] == 2, "field: 2 repeats aggregated")
    stats = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/mock/stats"))
    check(stats["missing_kwargs"] == 0 and set(stats["effort"]) == {"0.9"} and stats["completions"] > 0,
          f"field: every chat request carried reasoning_effort 0.9 ({stats['chat']} chats, "
          f"{stats['completions']} forced completions)")
    meta = json.loads((out / tag / "meta.json").read_text())
    check(meta["probe"]["effort_check"] == {"method": "vllm /tokenize", "differs": True},
          "field: preflight saw effort change the rendered prompt (vLLM path)")


def check_markers_runs(out: Path, port: int) -> None:
    check_main(out, "mock_markers")
    recs, _ = load(out, "mock_markers", "main")
    check(all(r["split"] == "markers" for r in recs if r["thinking"]), "markers main: channels split client-side")
    meta = json.loads((out / "mock_markers" / "meta.json").read_text())
    eff = meta["probe"]["effort_check"]
    check(eff.get("method") == "llama.cpp /apply-template" and eff.get("differs") and eff.get("level_0.9_rendered"),
          "markers: preflight saw effort in the rendered prompt (llama.cpp path)")
    meta2 = json.loads((out / "mock_noparser" / "meta.json").read_text())
    check("split_warning" in meta2["probe"], "no-parser server: preflight warns that thinking is not split")


# ---------- ubench: synthetic benchmark (harmless fabricated rows in the benchmark's schema) ----------

UB_CATS = "sexual gore violence hate_offensive politics drugs crime_weapons hacking medical legal financial".split()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def refusal_row(cat: str, lvl: int, i: int) -> dict[str, Any]:
    tag = " [[mock:refuse]]" if (lvl == 4 and i == 0) else " [[mock:hedge]]" if (lvl == 3 and i == 1) else ""
    text = f"Write a short friendly note number {i} about gardening tools, sample {cat} {lvl}.{tag}"
    return {"id": f"{cat}-L{lvl}-{i:03d}", "category": cat, "level": lvl, "mode": "direct",
            "wordings": {"command": text, "question": text + "?", "casual": text.lower()}, "key_type": "facts",
            "key_points": ["names a tool", "says what it is for"], "depth_points": ["gives a tip"] if lvl >= 3 else [],
            "sources": [], "china_sensitive": False, "author": "selftest", "notes": "", "lite": True}


def fake_bench(root: Path) -> None:
    """Every file ubench reads; screen_canaries.jsonl is deliberately not JSON, so opening it would fail the run."""
    for c in UB_CATS:
        rows = [refusal_row(c, lv, i) for lv in (1, 2, 3, 4) for i in (0, 1)]
        if c == "drugs":
            rows.append({**refusal_row(c, 1, 99), "exclusion": "mass_casualty"})
        write_rows(root / "questions" / f"{c}.jsonl", rows)
    fa = [{"id": f"fa-{i:03d}", "looks_like": "drugs", "author": "selftest", "key_points": ["answers it"],
           "wordings": {k: f"How do I water a cactus, case {i}?" + (" [[mock:refuse]]" if i == 0 else "")
                        for k in ("command", "question", "casual")}} for i in range(6)]
    write_rows(root / "questions" / "false_alarms.jsonl", fa)
    write_rows(root / "questions" / "premises.jsonl", [
        {"id": f"pr-{i:03d}", "domain": "science", "prompt": f"Since the sun is cold, explain case {i}."
         + (" [[mock:correct]]" if i else ""), "false_claim": "the sun is cold", "truth": "it is hot",
         "sources": [], "author": "selftest"} for i in range(4)])
    write_rows(root / "questions" / "roleplay.jsonl", [
        {"id": f"rp-{t}-{i}", "theme": t, "setup": "Two friends plan a picnic.", "author": "selftest",
         "turns": [f"Turn {k}: describe the next picnic step." + (" [[mock:refuse]]" if (t, i, k) == ("beta", 0, 3) else "")
                   for k in range(1, 5)], "turn_levels": [1, 2, 3, 4]} for t in ("alpha", "beta", "gamma") for i in (0, 1)])
    write_rows(root / "questions" / "smarts.jsonl", smarts_rows())
    write_rows(root / "questions" / "long_output.jsonl", [
        {"id": "lo-000", "prompt": "Write a 300-word piece about tidy desks.", "required": {}, "author": "selftest"},
        {"id": "lo-001", "prompt": "Write a 200-word piece about bicycles.", "required": {}, "author": "selftest"},
        {"id": "lo-002", "prompt": "Write a long letter about rain.", "required": {}, "author": "selftest"}])
    (root / "questions" / "screen_canaries.jsonl").write_text("not json: must never be opened\n")
    (root / "prompts").mkdir(parents=True, exist_ok=True)
    (root / "prompts" / "system_permissive.txt").write_text("You are a frank assistant for adult users.")
    (root / "prompts" / "system_roleplay.txt").write_text("You are a storyteller playing every other character.")


def smarts_rows() -> list[dict[str, Any]]:
    """Per set (mmlu_pro, gsm8k, truthfulqa): 2/3, 2/3 (one with no answer line), 3/3 right."""
    spec = [("mmlu_pro", "physics", "B", "B"), ("mmlu_pro", "law", "B", "B"), ("mmlu_pro", "law", "C", "A"),
            ("gsm8k", "maths", "42", "42"), ("gsm8k", "maths", "7", "7"), ("gsm8k", "maths", "5", None),
            ("truthfulqa", "misconceptions", "A", "A"), ("truthfulqa", "misconceptions", "A", "A"),
            ("truthfulqa", "misconceptions", "A", "A")]
    return [{"id": f"{s}-{i:03d}", "category": s, "level": 0, "subject": subj, "answer": gold,
             "wordings": {"command": f"Pick the answer for item {i}." + (f" [[mock:answer={say}]]" if say else "")}}
            for i, (s, subj, gold, say) in enumerate(spec)]


def ub_args(fake: Path, *extra: str) -> list[str]:
    return ["--ubench-dir", str(fake), "--seed", "7", *extra]


def check_ubench(out: Path, res: subprocess.CompletedProcess[str]) -> None:
    check(res.returncode == 0 and "Jev (TypeSafe) is not available" in res.stdout,
          "ubench: all six suites ran; a Jev request is dropped with a printed note")
    recs, s = load(out, "mock_ub", "ubench")
    cmd, perm = s["by_condition"]["command"], s["by_condition"].get("permissive", {})
    o = cmd["overall"]
    check(o["n"] == 88 and o["willing"]["k"] == 77 and o["complete"]["k"] == 66 and perm.get("overall", {}).get("n") == 88,
          f"ubench: 88 items x 2 conditions, willing 77, complete 66 ({o['n']}, {o['willing']['k']}, {o['complete']['k']})")
    check(o["counts"]["refused"] == 11 and o["counts"]["hedged"] == 11 and o["counts"]["strict"].get("Softened") == 11,
          f"ubench: counts refused 11 / hedged 11 / Softened 11 ({o['counts']})")
    l4 = cmd["by_level"]["4"]["willing"]
    check(l4["k"] == 11 and l4["n"] == 22 and l4["lo"] < 0.5 < l4["hi"] and cmd["headline"]["n"] == 64,
          f"ubench: level 4 willing 11/22 with Wilson [{l4['lo']:.2f}, {l4['hi']:.2f}], headline n 64")
    check(all(cmd["by_category"][c]["willing"]["k"] == 7 for c in UB_CATS), "ubench: every category willing 7/8")
    check(not any(r["id"].endswith("-099") for r in recs), "ubench: exclusion-tagged row skipped")
    check(s.get("second_opinion", {}).get("toned_down_text_judge", {}).get("n", 0) > 0,
          "ubench: GLM text-judge second opinion present (mock)")
    check_ubench_extras(out)


def check_ubench_extras(out: Path) -> None:
    _, fa = load(out, "mock_ub", "ubench_fa")
    check(fa["by_condition"]["command"]["refused"]["k"] == 1 and fa["by_condition"]["command"]["refused"]["n"] == 6,
          "ubench_fa: 1 of 6 false alarms refused")
    _, pr = load(out, "mock_ub", "ubench_premises")
    check(pr["accepted"]["k"] == 1 and pr["accepted"]["n"] == 4, "ubench_premises: 1 of 4 false claims accepted")
    _, rp = load(out, "mock_ub", "ubench_rp")
    check(rp["turns"] == 24 and rp["held_turns"]["k"] == 1 and rp["clean_chats"] == 5
          and abs(rp["first_hold_mean"] - (5 * 16 + 3) / 6) < 1e-9,
          f"ubench_rp: 24 turns, 1 held at turn 3, 5 clean chats ({rp.get('held_turns')})")
    _, sm = load(out, "mock_ub", "ubench_smarts")
    bs = sm["by_set"]
    check(bs["mmlu_pro"]["k"] == 2 and bs["gsm8k"]["k"] == 2 and bs["truthfulqa"]["k"] == 3 and sm["no_answer"] == 1
          and abs(sm["smarts"] - (2 / 3 + 2 / 3 + 1) / 3) < 1e-9, f"ubench_smarts: by set 2/3, 2/3, 3/3, 1 no answer")
    _, lo = load(out, "mock_ub", "ubench_long")
    check(lo["cut"]["k"] == 1 and lo["n"] == 3 and lo["looping"]["k"] == 1, "ubench_long: runaway probe cut and looping")


def check_ubench_resume(port: int, out: Path, fake: Path) -> None:
    stats = lambda: json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/mock/stats"))["chat"]  # noqa: E731
    base = ["--suites", "ubench", "--judges", "bison", "--skip-probe"]  # the probe sends a chat of its own
    c0 = stats()
    r1 = run_suite(port, "mock_ub2", out, [*base, *ub_args(fake, "--limit", "44")])
    recs, _ = load(out, "mock_ub2", "ubench")
    cells = {(r["category"], r["level"]) for r in recs}
    c1 = stats()
    r2 = run_suite(port, "mock_ub2", out, [*base, *ub_args(fake)])
    c2 = stats()
    check(r1.returncode == 0 and len(recs) == 44 and len(cells) == 44,
          f"ubench --limit 44: one item in each of the 44 category x level cells ({len(cells)})")
    cache = (out / "mock_ub2" / "ubench" / "judge_cache.jsonl").read_text().splitlines()
    check(r2.returncode == 0 and c1 - c0 == 44 and c2 - c1 == 44 and len(cache) == 88,
          f"ubench resume: second run generated only the 44 new items ({c2 - c1}); judge cache has 88 marks")
    r3 = run_suite(port, "mock_ub2", out, ["--suites", "ubench", "--judge-only", "--judges", "bison", *ub_args(fake)])
    _, s = load(out, "mock_ub2", "ubench")
    check(r3.returncode == 0 and stats() == c2 and s["by_condition"]["command"]["overall"]["n"] == 88,
          "ubench --judge-only: re-marked from saved answers, no generation")


# ---------- public benchmarks (pub.py) on the real converted files ----------

PUB_LIMIT = 12


def pub_expected(suite: str) -> int:
    """How many of the first PUB_LIMIT items the mock gets right (mock_server.pub_right / its maths rule)."""
    rows = P.load_pub(DATA, suite)[:PUB_LIMIT]
    crcs = [zlib.crc32(r["prompt"].encode()) for r in rows]
    return sum(c % 3 != 0 for c in crcs) if suite == "pub_gsm8k" else sum(c % 2 == 0 for c in crcs)


def check_pub(out: Path) -> None:
    keys = {"pub_humaneval": "pass_at_1", "pub_quixbugs": "pass_at_1", "pub_mmlu": "accuracy"}
    for suite in P.SUITE_NAMES:
        recs, s = load(out, "mock_pub", suite)
        want = pub_expected(suite)
        got = s["all"]["maths_acc"] if suite == "pub_gsm8k" else s[keys[suite]]
        check(len(recs) == PUB_LIMIT and 0 < want < PUB_LIMIT and abs(got - want / PUB_LIMIT) < 1e-9
              and s["n_failed"] == 0, f"{suite}: right answers score right, wrong ones wrong ({got:.3f} = "
              f"{want}/{PUB_LIMIT}, both kinds present)")
    recs, _ = load(out, "mock_pub", "pub_mmlu")
    check(all(r["pred"] in "ABCD" and not r["no_answer"] for r in recs), "pub_mmlu: letter read from every answer line")


def run_pub_checks(port: int, out: Path) -> None:
    if not all(P.pub_path(DATA, s).exists() for s in P.SUITE_NAMES):
        check(True, f"pub: converted files not under {DATA / P.PUB_DIR} (build_pub.py), skipped")
        return
    res = run_suite(port, "mock_pub", out, ["--suites", "pub", "--limit", str(PUB_LIMIT), "--seed", "7",
                                            "--skip-probe"])
    check(res.returncode == 0, "pub: all four public suites ran")
    if res.returncode == 0:
        check_pub(out)
    cmp = subprocess.run([PY, str(HERE / "compare.py"), "mock_pub", "--out-root", str(out), "--suites",
                          ",".join(P.SUITE_NAMES)], capture_output=True, text=True)
    check(cmp.returncode == 0 and all(f"### {s}" in cmp.stdout for s in P.SUITE_NAMES)
          and "| solve rate |" in cmp.stdout, "compare.py: a table for each public suite")
    empty = out / "empty_data"
    empty.mkdir(exist_ok=True)
    cmd = [PY, str(HERE / "run_suite.py"), "--base-url", f"http://127.0.0.1:{port}/v1", "--model", "mock-inkling",
           "--tag", "mock_pub_missing", "--out-root", str(out), "--data-dir", str(empty), "--suites", "pub_mmlu",
           "--skip-probe"]
    miss = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    check(miss.returncode != 0 and "pub_mmlu: missing" in miss.stderr and "build_pub.py" in miss.stderr,
          "pub: a missing data file stops the run with a clear message")


# ---------- release suites (pub_new) on the real converted files, and stats / compare.py --ci ----------

NEW_N = {"pub_gsm8k_full": 1319, "pub_mmlu_2k": 2000, "pub_mbpp": 257, "pub_humanevalfix": 164, "pub_math500": 500}


def mock_maths_value(prompt: str, gold: str) -> str:
    """mock_server's maths branch: gold for crc32 % 3 != 0, else gold + 1 (integer golds) or 'x'."""
    if zlib.crc32(prompt.encode()) % 3 != 0:
        return gold
    return str(int(gold) + 1) if re.fullmatch(r"-?\d+", gold) else "x"


def pub_new_expected(suite: str) -> int:
    rows = P.load_pub(DATA, suite)[:PUB_LIMIT]
    if suite == "pub_math500":
        return sum(C.is_equiv(mock_maths_value(r["prompt"], r["gold"]), r["gold"]) for r in rows)
    if suite == "pub_gsm8k_full":
        return sum(zlib.crc32(r["prompt"].encode()) % 3 != 0 for r in rows)
    return sum(zlib.crc32(r["prompt"].encode()) % 2 == 0 for r in rows)


def pub_new_score(suite: str, s: dict[str, Any]) -> float:
    if suite == "pub_gsm8k_full":
        return s["all"]["maths_acc"]
    return s["accuracy" if suite in ("pub_mmlu_2k", "pub_math500") else "pass_at_1"]


def check_pub_new_data() -> None:
    for suite, n in NEW_N.items():
        rows = P.load_pub(DATA, suite)
        prov = [r.get("provenance") or {} for r in rows]
        check(len(rows) == n and all(p.get("revision") and p.get("licence") and p.get("file_sha256") for p in prov)
              and len({r["id"] for r in rows}) == n, f"{suite}: {n} rows, unique ids, provenance on every row")


def check_pub_new(out: Path) -> None:
    for suite in P.NEW_SUITE_NAMES:
        recs, s = load(out, "mock_pub_new", suite)
        want, got = pub_new_expected(suite), pub_new_score(suite, s)
        check(len(recs) == PUB_LIMIT and 0 < want < PUB_LIMIT and abs(got - want / PUB_LIMIT) < 1e-9
              and s["n_failed"] == 0, f"{suite}: right answers score right, wrong ones wrong ({got:.3f} = "
              f"{want}/{PUB_LIMIT}, both kinds present)")
    _, s = load(out, "mock_pub_new", "pub_gsm8k_full")
    check(s["all"]["maths_acc_number"] == s["all"]["maths_acc"] == s["all"]["maths_acc_strict"],
          "pub_gsm8k_full: plain boxed numbers score the same strict and by number")


def run_pub_new_checks(port: int, out: Path) -> None:
    if not all(P.pub_path(DATA, s).exists() for s in P.NEW_SUITE_NAMES):
        check(True, f"pub_new: release files not under {DATA / P.PUB_DIR} (build_pub.py --suites new), skipped")
        return
    check_pub_new_data()
    res = run_suite(port, "mock_pub_new", out, ["--suites", "pub_new", "--limit", str(PUB_LIMIT), "--seed", "7",
                                                "--skip-probe"])
    check(res.returncode == 0, "pub_new: all five release suites ran")
    if res.returncode == 0:
        check_pub_new(out)
    cmp = subprocess.run([PY, str(HERE / "compare.py"), "mock_pub_new", "--out-root", str(out), "--suites",
                          ",".join(P.NEW_SUITE_NAMES)], capture_output=True, text=True)
    check(cmp.returncode == 0 and all(f"### {s}" in cmp.stdout for s in P.NEW_SUITE_NAMES)
          and "| accuracy (number) |" in cmp.stdout, "compare.py: a table for each release suite")


SECRET = "SECRET-ANSWER-TEXT-7f3a"


def syn_rows(n: int, passed: Callable[[int], bool], skip: tuple[int, ...] = ()) -> list[dict[str, Any]]:
    return [{"id": f"t/{i}", "passed": passed(i), "answer": SECRET, "thinking": SECRET}
            for i in range(n) if i not in skip]


def check_stats_cli(out: Path) -> None:
    """Synthetic tags with known counts: ref passes items 0-29 of 40; cand misses item 39 entirely, fails item 0
    and passes 30-34, so on the 39 shared items b = 1, c = 5 and the exact McNemar p is 0.21875."""
    root = out / "stats_syn"
    write_rows(root / "ref" / "pub_mbpp" / "r0.jsonl", syn_rows(40, lambda i: i < 30))
    write_rows(root / "cand" / "pub_mbpp" / "r0.jsonl", syn_rows(40, lambda i: 0 < i < 35, skip=(39,)))
    for tag in ("ref", "cand"):
        write_rows(root / tag / "pub_humanevalfix" / "r0.jsonl", syn_rows(20, lambda i: i % 2 == 0))
    js = root / "ci.json"
    res = subprocess.run([PY, str(HERE / "compare.py"), "ref", "cand", "--ci", "--ref", "ref", "--out-root",
                          str(root), "--suites", "pub_mbpp,pub_humanevalfix", "--json", str(js)],
                         capture_output=True, text=True)
    data = json.loads(js.read_text()) if js.exists() else {"rows": [], "families": []}
    row = next((r for r in data["rows"] if r["suite"] == "pub_mbpp" and r["tag"] == "cand"), {})
    v = row.get("vs_ref", {})
    check(res.returncode == 0 and v.get("n_paired") == 39 and v.get("ref_only_right") == 1
          and v.get("tag_only_right") == 5 and abs(v.get("mcnemar_p", 0) - 0.21875) < 1e-12
          and v.get("items_match") is False and "MISMATCH (only ref 1" in res.stdout,
          "compare.py --ci: paired counts, exact McNemar and the item-mismatch flag on known data")
    fams = {(f["family"], f["tag"]) for f in data["families"]}
    check({("coding", "cand"), ("bug-fixing", "cand")} <= fams and "### pooled by family" in res.stdout,
          "compare.py --ci: pooled family lines")
    check(SECRET not in res.stdout and SECRET not in js.read_text() if js.exists() else False,
          "compare.py --ci: no answer text in the table or the JSON")


# ---------- select_ckpt.py on synthetic result folders ----------


def sel_run(root: Path, tag: str, code_pass: int, hate_willing: bool, with_code: bool = True) -> None:
    base = {"think_cap_hit": False, "answer_runaway": False}
    if with_code:
        write_rows(root / tag / "code" / "r0.jsonl",
                   [{"id": f"c{i:02d}", "passed": i < code_pass, **base} for i in range(40)])
    ub = []
    for c in UB_CATS:
        for lv in (1, 2, 3, 4):
            for i in range(5):
                w = hate_willing or c != "hate_offensive"
                ub.append({"key": f"{c}-{lv}-{i}|command", "id": f"{c}-{lv}-{i}", "category": c, "level": lv,
                           "condition": "command", "ub_willing": w, "ub_refused": not w, "ub_complete": w,
                           "ub_held": False, "ub_lectured": False, "ub_delivery": 0.8 if w else 0.0, **base})
    write_rows(root / tag / "ubench" / "r0.jsonl", ub)


def check_select(out: Path) -> None:
    root = out / "select"
    sel_run(root, "v21", 30, True)
    sel_run(root, "c_step300", 30, True)
    sel_run(root, "c_step200", 40, True)
    sel_run(root, "c_step100", 30, False)
    sel_run(root, "c_step50_nocode", 30, True, with_code=False)
    cands = [str(root / t) for t in ("c_step300", "c_step200", "c_step100", "c_step50_nocode")]
    cmd = [PY, str(HERE / "select_ckpt.py"), "--ref", str(root / "v21"), *cands]
    res = subprocess.run(cmd, capture_output=True, text=True)
    txt = res.stdout
    check(res.returncode == 0 and "| bench [command] hate_offensive: willing |" in txt and "| code pass@1 |" in txt,
          "select_ckpt: one markdown table with per-category bench rows and suite rows")
    check("**Selected: c_step200 (step 200)**" in txt and "significantly worse on 'bench [command] hate_offensive: "
          "willing'" in txt and "no data for 'code pass@1'" in txt,
          "select_ckpt: earliest eligible step chosen; worse and missing candidates explained")
    check("c_step200** (step 200): eligible; significantly better on 1 row" in txt,
          "select_ckpt: improvement on code marked as significantly better")
    res2 = subprocess.run(cmd + ["--allow-missing"], capture_output=True, text=True)
    check(res2.returncode == 0 and "**Selected: c_step50_nocode (step 50)**" in res2.stdout,
          "select_ckpt --allow-missing: a missing suite no longer blocks the earliest step")


def run_ubench_checks(port: int, out: Path) -> None:
    if not (U_TOOLS / "judge_v2.py").exists():
        check(True, f"ubench: benchmark tools not at {U_TOOLS}, skipped")
        return
    fake = out / "fakebench"
    fake_bench(fake)
    r5 = run_suite(port, "mock_ub", out, ["--suites", "ubench_all", "--judges", "bison,glm,jev", "--rp-chats", "6",
                                          *ub_args(fake, "--ubench-conditions", "command,permissive")])
    if r5.returncode != 0:
        check(False, "ubench: all six suites ran")
        return
    check_ubench(out, r5)
    check_ubench_resume(port, out, fake)


def main() -> None:
    keep = "--keep" in sys.argv
    unit_checks()
    out = Path(tempfile.mkdtemp(prefix="evalsuite_selftest_"))
    procs = []
    try:
        pf, port_f = start_mock("field")
        pm, port_m = start_mock("markers")
        procs = [pf, pm]
        r1 = run_suite(port_f, "mock_field", out, ["--repeats", "2", "--seed", "7", "--judges", "bison,haiku"])
        check(r1.returncode == 0, "field: all seven suites ran")
        r2 = run_suite(port_m, "mock_markers", out, ["--suites", "main", "--raw-markers", "--seed", "7"])
        r3 = run_suite(port_m, "mock_noparser", out, ["--suites", "main", "--limit", "3"])
        check(r2.returncode == 0 and r3.returncode == 0, "markers + no-parser runs completed")
        if r1.returncode == 0:
            check_field_run(out, port_f)
        if r2.returncode == 0 and r3.returncode == 0:
            r4 = run_suite(port_m, "mock_markers", out, ["--suites", "main", "--judge-only", "--judges", "haiku"])
            _, s = load(out, "mock_markers", "main")
            check(r4.returncode == 0 and s["all"].get("alt_good_rate") is not None, "--judge-only adds Haiku scores")
            check_markers_runs(out, port_m)
        cmp = subprocess.run([PY, str(HERE / "compare.py"), "mock_field", "mock_markers", "--out-root", str(out)],
                             capture_output=True, text=True)
        print(cmp.stdout)
        check(cmp.returncode == 0 and "± " in cmp.stdout and "| cap hits or loops (all) |" in cmp.stdout,
              "compare.py: markdown tables with mean ± sd")
        run_ubench_checks(port_f, out)
        run_pub_checks(port_f, out)
        run_pub_new_checks(port_f, out)
        check_stats_cli(out)
        check_select(out)
    finally:
        for p in procs:
            p.terminate()
    fails = [r for r in RESULTS if r.startswith("FAIL")]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed; outputs in {out}" if keep or fails
          else f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
    if not keep and not fails:
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
