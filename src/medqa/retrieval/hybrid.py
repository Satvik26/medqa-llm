"""Hybrid retriever: BM25 and dense rankings merged with reciprocal rank fusion (RRF)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from medqa.retrieval.dense import DenseIndex
from medqa.retrieval.sparse import BM25Index


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = 60
) -> list[tuple[int, float]]:
    """score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


@dataclass
class Passage:
    doc_id: int
    text: str
    answer: str
    score: float


class HybridRetriever:
    """Index over flashcards: `corpus` has columns `id`, `question`, `answer`."""

    def __init__(self, corpus: pd.DataFrame, sparse: BM25Index, dense: DenseIndex) -> None:
        self.corpus = corpus.reset_index(drop=True)
        self.sparse = sparse
        self.dense = dense

    @staticmethod
    def passage_text(question: str, answer: str) -> str:
        return f"{question.strip()} {answer.strip()}"

    @classmethod
    def build(cls, corpus: pd.DataFrame, model_name: str) -> HybridRetriever:
        texts = [
            cls.passage_text(q, a)
            for q, a in zip(corpus["question"], corpus["answer"], strict=True)
        ]
        return cls(corpus, BM25Index.build(texts), DenseIndex.build(texts, model_name))

    def search(
        self,
        queries: Sequence[str],
        k: int = 3,
        exclude: Sequence[set[int]] | None = None,
        candidates: int = 30,
    ) -> list[list[Passage]]:
        """Top-k passages per query; `exclude[i]` holds doc ids that query i must not see."""
        sparse_pos, _ = self.sparse.search(queries, candidates)
        dense_pos, _ = self.dense.search(queries, candidates)
        ids = self.corpus["id"].to_numpy()
        results = []
        for i in range(len(queries)):
            banned = exclude[i] if exclude else set()
            fused = reciprocal_rank_fusion([sparse_pos[i].tolist(), dense_pos[i].tolist()])
            hits = []
            for pos, score in fused:
                if int(ids[pos]) in banned:
                    continue
                row = self.corpus.iloc[pos]
                hits.append(
                    Passage(
                        int(row["id"]),
                        self.passage_text(row["question"], row["answer"]),
                        row["answer"],
                        score,
                    )
                )
                if len(hits) == k:
                    break
            results.append(hits)
        return results

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.corpus.to_parquet(path / "corpus.parquet", index=False)
        self.sparse.save(path / "bm25")
        self.dense.save(path / "dense")

    @classmethod
    def load(cls, path: Path) -> HybridRetriever:
        if not (path / "corpus.parquet").exists():
            raise FileNotFoundError(f"No index at {path}. Run `medqa rag build-index` first.")
        return cls(
            pd.read_parquet(path / "corpus.parquet"),
            BM25Index.load(path / "bm25"),
            DenseIndex.load(path / "dense"),
        )
