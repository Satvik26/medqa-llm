"""Chat-format prompts shared by training, inference, the frontier baseline and the demo.

Every model is prompted through its own chat template, so the zero-shot baselines, the fine-tuned
models and the API model all see exactly the same messages.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

SYSTEM_PROMPT = (
    "You are a medical tutor. Answer the question accurately and concisely, "
    "in one to three sentences, like a medical flashcard."
)

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def build_user_message(question: str, contexts: Sequence[str] | None = None) -> str:
    question = question.strip()
    if not contexts:
        return question
    passages = "\n".join(f"[{i}] {c.strip()}" for i, c in enumerate(contexts, start=1))
    return f"Context (may be incomplete or irrelevant):\n{passages}\n\nQuestion: {question}"


def build_messages(
    question: str,
    contexts: Sequence[str] | None = None,
    answer: str | None = None,
    system: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_message(question, contexts)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages


def _apply(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    # `enable_thinking` switches off Qwen3 / SmolLM3 reasoning traces; other templates ignore it.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def render_prompt(tokenizer: Any, question: str, contexts: Sequence[str] | None = None) -> str:
    """Prompt text ending with the assistant header, ready for generation."""
    return _apply(tokenizer, build_messages(question, contexts), add_generation_prompt=True)


def render_example(
    tokenizer: Any, question: str, answer: str, contexts: Sequence[str] | None = None
) -> str:
    """Full training conversation including the reference answer and end-of-turn tokens."""
    return _apply(
        tokenizer, build_messages(question, contexts, answer), add_generation_prompt=False
    )


def clean_generation(text: str, stop_tokens: Sequence[str] = ()) -> str:
    """Remove reasoning traces and anything after an end-of-turn token."""
    text = _THINK_RE.sub("", text)
    cut = min((i for t in stop_tokens if t and (i := text.find(t)) != -1), default=len(text))
    return text[:cut].strip()
