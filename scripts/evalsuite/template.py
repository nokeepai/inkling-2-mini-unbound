"""Text-only Python port of Inkling-2-Mini's official chat template (out/final/configs/chat_template.jinja).

Used only where the suite must build a raw prompt itself: the forced answer after a thinking-cap hit
(/v1/completions continuation) and the local fallback for prompt-token counting. Normal requests go
through the server's own template via /v1/chat/completions.
"""
from __future__ import annotations

from typing import Any

EFFORT_MAP = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}
ROLE_TOKEN = {"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>",
              "tool": "<|message_tool|>"}
MESSAGE_MODEL, CONTENT_THINKING, CONTENT_TEXT = "<|message_model|>", "<|content_thinking|>", "<|content_text|>"
END_MESSAGE, END_SAMPLING = "<|end_message|>", "<|content_model_end_sampling|>"


def effort_message(effort: float | str | None) -> str:
    eff: float | str = 0.9 if effort is None else effort
    num = EFFORT_MAP[eff.strip()] if isinstance(eff, str) else float(eff)
    if not 0.0 <= num <= 0.99:
        raise ValueError("reasoning_effort must be in [0.0, 0.99]")
    # jinja prints floats with str(); 0.0 is special-cased to "0" in the template.
    shown = "0" if num == 0.0 else str(num)
    return f"<|message_system|>{CONTENT_TEXT}Thinking effort level: {shown}{END_MESSAGE}"


def render_message(m: dict[str, Any]) -> str:
    role = m["role"]
    rtok = ROLE_TOKEN[role]
    content = m.get("content") or ""
    reasoning = m.get("reasoning_content") or ""
    out = ""
    if role == "assistant" and reasoning:
        out += rtok + CONTENT_THINKING + reasoning + END_MESSAGE
    if content or not (role == "assistant" and reasoning):
        out += rtok + CONTENT_TEXT + content + END_MESSAGE
    if role == "assistant":
        out += END_SAMPLING
    return out


def render(messages: list[dict[str, Any]], effort: float | str | None) -> str:
    """The effort system message goes after leading system messages, before the first other one."""
    parts: list[str] = []
    emitted = False
    for m in messages:
        if not emitted and m["role"] != "system":
            parts.append(effort_message(effort))
            emitted = True
        parts.append(render_message(m))
    if not emitted:
        parts.append(effort_message(effort))
    return "".join(parts)


def forced_answer_prompt(messages: list[dict[str, Any]], effort: float, thinking: str) -> str:
    """Prompt + the capped thinking, closed, with the answer channel opened (the lab's forced tokens:
    <|end_message|><|message_model|><|content_text|> after the thinking)."""
    return (render(messages, effort) + MESSAGE_MODEL + CONTENT_THINKING + thinking
            + END_MESSAGE + MESSAGE_MODEL + CONTENT_TEXT)
