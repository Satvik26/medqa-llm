"""Load Medical Meadow flashcards into a tidy `question` / `answer` DataFrame."""

from __future__ import annotations

import pandas as pd


def clean_flashcards(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the Alpaca columns, drop empty rows and exact duplicate pairs."""
    out = df.rename(columns={"input": "question", "output": "answer"})[["question", "answer"]]
    out = out.astype(str).apply(lambda col: col.str.strip())
    out = out[(out["question"] != "") & (out["answer"] != "")]
    out = out.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
    out.insert(0, "id", range(len(out)))
    return out


def load_flashcards(dataset: str = "medalpaca/medical_meadow_medical_flashcards") -> pd.DataFrame:
    from datasets import load_dataset

    raw = load_dataset(dataset, split="train").to_pandas()
    return clean_flashcards(raw)
