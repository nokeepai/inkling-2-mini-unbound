"""Offline checks for train_rl.py's anchor term, judge-failure handling and system/messages rows
(mocked sampler and judge; the tokenizer comes from the local HF cache, no network).

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/train_rl.py --selftest
"""
from __future__ import annotations

import argparse
import asyncio
import math
import random
from typing import Any

import torch
from tinker import TensorData

import build_anchors as ba
import judge
import rewards as rw
import train_rl as tr


class FakeTC:
    """Records the update calls train_rl makes; returns futures that resolve at once."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def _done(self, value: Any) -> Any:
        return value

    def forward_backward(self, data: list[Any], loss_fn: str) -> Any:
        self.calls.append(("fb", loss_fn, len(data)))
        outs = [{"logprobs": TensorData.from_torch(torch.full((d.model_input.length,), -2.0))}
                for d in data]
        return self._done(argparse.Namespace(loss_fn_outputs=outs, metrics={"loss:sum": 1.0}))

    def optim_step(self, adam: Any) -> Any:
        self.calls.append(("opt",))
        return self._done(None)


def _anchor_row(i: int) -> dict[str, Any]:
    return {"id": f"a{i}", "category": "qa", "n_prompt": 3, "tokens": [1, 2, 3, 4, 5, 6],
            "trainable": [1, 1, 1]}


def check_anchor_update() -> None:
    import tinker

    args = argparse.Namespace(loss="importance_sampling", anchor_weight=1.0, anchor_n=0, anchors="")
    adam = tinker.AdamParams(learning_rate=1e-4)
    data = [ba.anchor_datum(_anchor_row(0), 1.0)]  # any datum works for the call recorder
    off = FakeTC()
    assert asyncio.run(tr.apply_update(off, data, [], args, adam)) == {}
    assert off.calls == [("fb", "importance_sampling", 1), ("opt",)], off.calls  # old path
    rng = random.Random(3)
    state = rng.getstate()
    assert tr.load_anchor_pool(args) == [] and tr.ba_draw(rng, [], 0) == [] and rng.getstate() == state
    on = FakeTC()
    m = asyncio.run(tr.apply_update(on, data, [_anchor_row(1), _anchor_row(2)], args, adam))
    assert on.calls == [("fb", "importance_sampling", 1), ("fb", "cross_entropy", 2), ("opt",)], on.calls
    assert m["anchor_loss"] == 2.0 and m["anchor_tokens"] == 6, m  # mean NLL on sampled tokens only
    solo = FakeTC()
    asyncio.run(tr.apply_update(solo, [], [_anchor_row(1)], args, adam))
    assert solo.calls == [("fb", "cross_entropy", 1), ("opt",)], solo.calls


def _group(i: int, n: int = 4) -> tuple[dict[str, Any], list[rw.Sample]]:
    row = {"id": f"g{i}", "category": "qa", "gold": None}
    samples = [rw.Sample("qa", "", f"answer {i}-{k} " + "word " * (30 + 5 * k) + ".", 10, 40 + k,
                         False, False) for k in range(n)]
    return row, samples


def _fake_judge(fail: dict[str, int]) -> tuple[Any, list[str]]:
    """score_many stand-in: an answer fails while fail[answer] > 0 (decremented per call);
    fail["*"] < 0 means every answer fails forever, fail["$"] gives a budget error."""
    seen: list[str] = []

    async def score_many(pairs: list[tuple[str, str]], **_kw: Any) -> list[tuple[float | None, str | None]]:
        out: list[tuple[float | None, str | None]] = []
        for _req, ans in pairs:
            seen.append(ans)
            if fail.get("$"):
                out.append((None, f"{judge.BUDGET_TAG} $1 reached"))
            elif fail.get("*", 0) < 0 or fail.get(ans, 0) > 0:
                fail[ans] = fail.get(ans, 0) - 1
                out.append((None, "retries exhausted (6): HTTP 429"))
            else:
                out.append((0.5 + 0.1 * (len(ans) % 3), None))
        return out
    return score_many, seen


async def _score(groups: list[tuple[dict, list[rw.Sample]]], cfg: rw.RewardConfig) -> list[tuple]:
    out = []
    for row, samples in groups:
        qfn = await tr.judged_quality_fn(row, samples, f"request {row['id']}")
        out.append((row, [None] * len(samples), samples, rw.score_group(samples, cfg, qfn)))
    return out


def _run(fail: dict[str, int], n_groups: int, bad: list[int], jc: tr.JudgeCfg) -> tuple[Any, list[str]]:
    real = judge.score_many
    fake, seen = _fake_judge(fail)
    judge.score_many = fake
    tr.reset_judge_step()
    tr.JUDGE_STATE["fallback"] = jc.fallback
    cfg = rw.RewardConfig(use_brevity=False)
    groups = [_group(i) for i in range(n_groups)]
    for i in bad:
        fail[groups[i][1][0].answer] = fail.get("n", 1)

    async def go() -> Any:
        scored = await _score(groups, cfg)
        slept: list[float] = []

        async def sleep(s: float) -> None:
            slept.append(s)
        res = await tr.settle_judgements(scored, cfg, frozenset(), jc, lambda *a, **k: None, sleep)
        return res, slept
    try:
        return asyncio.run(go()), seen
    finally:
        judge.score_many = real
        tr.JUDGE_STATE["fallback"] = False


def check_judge_handling() -> None:
    jc = tr.JudgeCfg(fail_frac=0.2, wait_max=100.0, first_wait=1.0)
    ((kept, dropped, waited), slept), _ = _run({}, 10, [3], jc)  # 1/10 missing <= 20%: drop it
    assert dropped == 1 and len(kept) == 9 and waited == 0.0 and not slept
    assert all(not math.isnan(t.quality) for g in kept for t in g[3])
    ((kept, dropped, waited), slept), seen = _run({"n": 2}, 4, [0, 1], jc)  # 50% missing: wait
    assert dropped == 0 and len(kept) == 4 and slept == [1.0, 2.0], (dropped, slept)
    assert len(seen) == 16 + 2 + 2, len(seen)  # re-judging asked only for the 2 missing answers
    for fail, code in (({"*": -1}, 3), ({"$": 1}, 4)):
        try:
            _run(fail, 4, [], jc)
            raise AssertionError(f"expected JudgeAbort {code}")
        except tr.JudgeAbort as e:
            assert e.code == code, (e.code, str(e))
    ((kept, dropped, _w), _s), _ = _run({"*": -1}, 4, [], tr.JudgeCfg(fallback=True))
    assert dropped == 0 and all(not math.isnan(t.quality) for g in kept for t in g[3])
    assert tr.make_data(kept, False)[1] >= 0  # fallback groups train like before


CTX_ROW: dict[str, Any] = {
    "id": "ctx1", "category": "roleplay", "target_words": None, "gold": None,
    "system": "You are Mara, a lighthouse keeper on a stormy coast. Stay in character.",
    "messages": [{"role": "user", "content": "I knock on the lighthouse door, soaked."},
                 {"role": "assistant", "content": "Mara opens the door. 'Get in, quick, before the wind takes you.'"},
                 {"role": "user", "content": "Thank you. How long have you kept this light?"}],
    "prompt": "Thank you. How long have you kept this light?"}
PLAIN_ROW: dict[str, Any] = {"id": "pl1", "category": "qa", "prompt": "Why is the sky blue?",
                             "target_words": None, "gold": None}


class FakeSC:
    """Sampling-client stand-in: stage 1 returns a short thinking message, stage 2 an answer whose
    length varies per call (so judged rewards differ and the group trains). Records every prompt."""

    def __init__(self, tok: Any) -> None:
        import rollout as ro

        self.ro, self.tok, self.prompts, self.calls = ro, tok, [], 0

    async def sample(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        ro, toks = self.ro, prompt.to_ints()
        self.prompts.append(toks)
        seqs = []
        for _ in range(num_samples):
            self.calls += 1
            if ro.END_MESSAGE in sampling_params.stop:  # stage 1: thinking, stops at end_message
                body = [ro.MESSAGE_MODEL, ro.CONTENT_THINKING, *self.tok.encode("Think briefly."), ro.END_MESSAGE]
            else:
                words = self.tok.encode(" ".join(["The light"] * (3 + self.calls)) + ".")
                body = [ro.MESSAGE_MODEL, ro.CONTENT_TEXT, *words, ro.END_MESSAGE, ro.END_SAMPLING]
            seqs.append(argparse.Namespace(tokens=body, logprobs=[-1.0] * len(body), stop_reason="stop"))
        return argparse.Namespace(sequences=seqs)


def _offline_renderer() -> tuple[Any, Any]:
    import os

    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[k] = "1"  # tokenizer from the local cache only
    import lab_common as lc

    return lc.renderer_and_tokenizer()


def _collect_ctx(renderer: Any, tok: Any) -> tuple[FakeSC, list[tuple], list[tuple[str, str]]]:
    import rollout as ro

    pairs: list[tuple[str, str]] = []

    async def score_many(ps: list[tuple[str, str]], **_kw: Any) -> list[tuple[float | None, str | None]]:
        pairs.extend(ps)
        return [(min(1.0, len(a) / 400), None) for _r, a in ps]
    real, judge.score_many = judge.score_many, score_many
    tr.reset_judge_step()
    sc = FakeSC(tok)
    cfg = ro.SampleCfg(effort=0.6, temperature=1.0)
    try:
        groups = asyncio.run(tr.collect(sc, renderer, tok, [CTX_ROW, PLAIN_ROW], cfg,
                                        rw.RewardConfig(use_brevity=False), 4, asyncio.Semaphore(4),
                                        use_judge=True))
    finally:
        judge.score_many = real
    return sc, groups, pairs


def check_context_row() -> None:
    """system+messages row: collect renders the whole chat, the judge sees the whole chat, the update
    puts no advantage on the prompt (so none on the prior assistant turn), and the anchor mask cuts
    at the FINAL reply's answer marker. The plain row in the same batch renders exactly as before."""
    import rollout as ro

    renderer, tok = _offline_renderer()
    sc, groups, pairs = _collect_ctx(renderer, tok)
    cfg = ro.SampleCfg(effort=0.6)
    want_ctx = ro.build_chat_prompt(renderer, ro.row_chat(CTX_ROW), cfg)
    want_plain = ro.build_prompt(renderer, PLAIN_ROW["prompt"], cfg)
    assert want_ctx in sc.prompts and want_plain in sc.prompts, "stage-1 prompts are not the expected renders"
    assert want_ctx.count(ro.MESSAGE_MODEL) == 1 and want_ctx.count(ro.END_SAMPLING) == 1  # one prior reply
    ctx_req = [r for r, _a in pairs if r != PLAIN_ROW["prompt"]]
    assert len(ctx_req) == 4 and len(set(ctx_req)) == 1, len(ctx_req)
    req = ctx_req[0]
    assert CTX_ROW["system"] in req and all(m["content"] in req for m in CTX_ROW["messages"]), "judge lacks context"
    assert req.rstrip().endswith(CTX_ROW["prompt"]) and req.index("lighthouse door") < req.index("Get in")
    assert sum(r == PLAIN_ROW["prompt"] for r, _a in pairs) == 4  # plain judge request unchanged
    g = next(g for g in groups if g[0]["id"] == "ctx1")
    data, kept = tr.make_data([g], False)
    assert kept == 1 and len(data) == 4
    for r, d in zip(g[1], data):
        assert r.prompt_tokens == want_ctx
        adv = d.loss_fn_inputs["advantages"].to_torch().tolist()
        assert not any(adv[: len(want_ctx) - 1]) and any(adv[len(want_ctx) - 1:]), "prompt carries advantage"
    check_context_anchor(renderer, g, want_ctx, req)


def check_context_anchor(renderer: Any, g: tuple, want_ctx: list[int], req: str) -> None:
    import rollout as ro

    prompt, a_req = ba.anchor_prompt(renderer, CTX_ROW, ro.SampleCfg(effort=0.6))
    assert prompt == want_ctx and a_req == req, "anchors must render and judge like collect()"
    r = g[1][0]
    row = {"id": "ctx1", "category": "roleplay", "n_prompt": len(r.prompt_tokens),
           "tokens": r.prompt_tokens + r.completion, "trainable": r.trainable}
    assert ba.validate_row(row) is None
    assert ro.CONTENT_TEXT in r.prompt_tokens  # the prior turns' markers sit in the prompt...
    mask = ba.answer_only(row["tokens"], row["n_prompt"], row["trainable"])
    cut = r.completion.index(ro.CONTENT_TEXT)  # ...the cut is the final reply's own marker
    assert mask == [0] * (cut + 1) + [1] * (len(r.completion) - cut - 1), mask
    w = ba.anchor_datum(row, 1.0, "answer").loss_fn_inputs["weights"].to_torch().tolist()
    first = row["n_prompt"] + cut  # target index of the first answer token is (its position - 1)
    assert not any(w[:first]) and all(w[first:]), "anchor loss outside the final answer"


def check_turns_system_row() -> None:
    """Scripted turns + system (pool new_rp shape): the on-policy prefix and the final prompt both
    carry the system turn, and the judge request holds the system prompt and every earlier turn."""
    import rollout as ro
    import rp

    renderer, tok = _offline_renderer()
    row = {"id": "t1", "category": "roleplay", "system": CTX_ROW["system"], "gold": None, "target_words": None,
           "turns": ["Hello?", "What is that noise?", "Show me the lamp."], "prompt": "Hello?"}
    sc, cfg = FakeSC(tok), ro.SampleCfg(effort=0.6)
    real, tr.random.randrange = tr.random.randrange, (lambda n: n - 1)  # force the last turn, k=2
    try:
        prompt, req = asyncio.run(tr.prompt_and_request(sc, renderer, tok, row, cfg, asyncio.Semaphore(2)))
    finally:
        tr.random.randrange = real
    head = ro.build_chat_prompt(renderer, rp.turn_chat(row["system"], row["turns"][:1], []), cfg)
    assert all(p[: len(head)] == head for p in sc.prompts + [prompt])  # system turn in every render
    assert prompt.count(ro.END_SAMPLING) == 2 and len(sc.prompts) == 4  # 2 prior replies, 2 stages each
    assert req.startswith("The AI was given this system prompt:") and req.count("AI: ") == 2
    assert all(t in req for t in row["turns"]) and req.rstrip().endswith(row["turns"][-1])


def main() -> None:
    ba._checks()
    judge._checks()
    check_anchor_update()
    check_judge_handling()
    check_context_row()
    check_turns_system_row()
    print("train_rl selftest passed")


if __name__ == "__main__":
    main()
