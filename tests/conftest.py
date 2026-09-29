from __future__ import annotations

import pandas as pd
import pytest


class FakeTokenizer:
    """Minimal stand-in for a Hugging Face tokenizer's chat template."""

    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def flashcards() -> pd.DataFrame:
    rows = [
        (f"What is condition {i}?", f"Condition {i} is a disorder of organ {i % 7}.")
        for i in range(200)
    ]
    return pd.DataFrame(
        {"id": range(len(rows)), "question": [q for q, _ in rows], "answer": [a for _, a in rows]}
    )
