"""BM25 keyword index (pure Python via `bm25s`, replacing the Java-based PyTerrier)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


class BM25Index:
    def __init__(self, retriever) -> None:
        self._retriever = retriever

    @classmethod
    def build(cls, corpus: Sequence[str]) -> BM25Index:
        import bm25s

        retriever = bm25s.BM25()
        retriever.index(
            bm25s.tokenize(list(corpus), stopwords="en", show_progress=False), show_progress=False
        )
        return cls(retriever)

    def search(self, queries: Sequence[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """Row-aligned (positions, scores), each of shape (len(queries), k)."""
        import bm25s

        tokens = bm25s.tokenize(list(queries), stopwords="en", show_progress=False)
        positions, scores = self._retriever.retrieve(tokens, k=k, show_progress=False)
        return np.asarray(positions), np.asarray(scores)

    def save(self, path: Path) -> None:
        self._retriever.save(str(path))

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        import bm25s

        return cls(bm25s.BM25.load(str(path)))
