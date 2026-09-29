"""Build the retrieval index and context-augmented splits for RAFT (retrieval-augmented fine-tuning).

Leakage rules:
- Only the training split is indexed, so validation/test answers can never be retrieved.
- A training question never retrieves its own flashcard, nor any flashcard with the identical
  answer; otherwise the model would learn to copy instead of to use context.
"""

from __future__ import annotations

import json

import pandas as pd

from medqa.config import ExperimentConfig
from medqa.data.split import load_splits, save_splits
from medqa.retrieval.hybrid import HybridRetriever


def build_index(cfg: ExperimentConfig) -> HybridRetriever:
    train = load_splits(cfg.data.split_dir)["train"]
    retriever = HybridRetriever.build(train[["id", "question", "answer"]], cfg.eval.semantic_model)
    retriever.save(cfg.data.index_dir)
    return retriever


def attach_contexts(
    frame: pd.DataFrame, retriever: HybridRetriever, k: int, exclude_self: bool
) -> pd.DataFrame:
    exclude = None
    if exclude_self:
        by_answer = retriever.corpus.groupby("answer")["id"].apply(set).to_dict()
        exclude = [
            {int(i)} | by_answer.get(a, set())
            for i, a in zip(frame["id"], frame["answer"], strict=True)
        ]
    hits = retriever.search(frame["question"].tolist(), k=k, exclude=exclude)
    out = frame.copy()
    out["contexts"] = [[h.text for h in row] for row in hits]
    out["context_ids"] = [[h.doc_id for h in row] for row in hits]
    # How often a retrieved flashcard already states the reference answer: exactly, or as a
    # paraphrase (the dataset contains many reworded copies of the same card).
    out["answer_in_context"] = [
        any(h.answer.strip() == a.strip() for h in row)
        for row, a in zip(hits, frame["answer"], strict=True)
    ]
    out["max_context_answer_cosine"] = _max_answer_similarity(retriever, frame["answer"], hits)
    out["near_duplicate_in_context"] = out["max_context_answer_cosine"] >= NEAR_DUPLICATE_COSINE
    return out


NEAR_DUPLICATE_COSINE = 0.9


def _max_answer_similarity(retriever: HybridRetriever, answers: pd.Series, hits) -> list[float]:
    """Cosine between each reference answer and the answers of its retrieved flashcards."""
    ref = retriever.dense.encode(answers.tolist())
    retrieved = retriever.dense.encode([h.answer for row in hits for h in row])
    sims, offset = [], 0
    for i, row in enumerate(hits):
        block = retrieved[offset : offset + len(row)]
        offset += len(row)
        sims.append(round(float((block @ ref[i]).max()), 4) if len(row) else 0.0)
    return sims


def build_rag_splits(cfg: ExperimentConfig) -> dict:
    retriever = HybridRetriever.load(cfg.data.index_dir)
    splits = load_splits(cfg.data.split_dir)
    augmented = {
        name: attach_contexts(frame, retriever, cfg.data.rag_top_k, exclude_self=(name == "train"))
        for name, frame in splits.items()
    }
    out_dir = cfg.data.split_dir.with_name(cfg.data.split_dir.name + "_rag")
    save_splits(augmented, out_dir)
    stats = {
        name: {
            "rows": len(f),
            "exact_answer_in_context_rate": round(float(f["answer_in_context"].mean()), 4),
            "near_duplicate_in_context_rate": round(
                float(f["near_duplicate_in_context"].mean()), 4
            ),
        }
        for name, f in augmented.items()
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
