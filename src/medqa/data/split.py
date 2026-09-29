"""Deterministic train / validation / test split, stored as Parquet so every run sees the same data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def make_splits(
    df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.05, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """`val_size` is a fraction of the whole dataset, carved out of what would be training data."""
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1")
    order = np.random.default_rng(seed).permutation(len(df))
    n_test = round(len(df) * test_size)
    n_val = round(len(df) * val_size)
    parts = {
        "test": order[:n_test],
        "val": order[n_test : n_test + n_val],
        "train": order[n_test + n_val :],
    }
    return {name: df.iloc[np.sort(idx)].reset_index(drop=True) for name, idx in parts.items()}


def overlap_stats(splits: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Share of test flashcards whose question or answer also appears verbatim in train.

    The dataset contains paraphrased duplicates, so this bounds how much of a test score can come
    from memorising the training set rather than generalising.
    """
    train, test = splits["train"], splits["test"]
    norm = lambda s: s.str.lower().str.strip()  # noqa: E731
    return {
        "test_question_in_train": float(
            norm(test["question"]).isin(set(norm(train["question"]))).mean()
        ),
        "test_answer_in_train": float(norm(test["answer"]).isin(set(norm(train["answer"]))).mean()),
    }


def save_splits(splits: dict[str, pd.DataFrame], split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(split_dir / f"{name}.parquet", index=False)


def load_splits(split_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [s for s in SPLITS if not (split_dir / f"{s}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing splits {missing} in {split_dir}. Run `medqa data split` first."
        )
    return {s: pd.read_parquet(split_dir / f"{s}.parquet") for s in SPLITS}
