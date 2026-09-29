"""Dense retrieval with a biomedical sentence-embedding model and exact cosine search."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import numpy as np


class DenseIndex:
    def __init__(self, model_name: str, embeddings: np.ndarray) -> None:
        self.model_name = model_name
        self.embeddings = embeddings

    @cached_property
    def encoder(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        vectors = self.encoder.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(vectors, dtype=np.float32)

    @classmethod
    def build(cls, corpus: Sequence[str], model_name: str) -> DenseIndex:
        index = cls(model_name, np.empty((0, 0), dtype=np.float32))
        index.embeddings = index.encode(corpus)
        return index

    def search(
        self, queries: Sequence[str], k: int, chunk: int = 512
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-k by cosine similarity (embeddings are L2-normalised, so a dot product)."""
        q = self.encode(queries)
        positions, scores = [], []
        for start in range(0, len(q), chunk):
            sims = q[start : start + chunk] @ self.embeddings.T
            top = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
            order = np.take_along_axis(sims, top, axis=1).argsort(axis=1)[:, ::-1]
            top = np.take_along_axis(top, order, axis=1)
            positions.append(top)
            scores.append(np.take_along_axis(sims, top, axis=1))
        return np.vstack(positions), np.vstack(scores)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        (path / "meta.json").write_text(json.dumps({"model_name": self.model_name}))

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        meta = json.loads((path / "meta.json").read_text())
        return cls(meta["model_name"], np.load(path / "embeddings.npy"))
