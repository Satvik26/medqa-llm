"""Answer-quality metrics. Every metric returns one score per example so we can bootstrap CIs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from medqa.config import EvalConfig


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Mean with a percentile-bootstrap (1 - alpha) confidence interval."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_resamples, arr.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi)}


def _nonempty(texts: Sequence[str]) -> list[str]:
    return [t if t.strip() else "(no answer)" for t in texts]


def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> np.ndarray:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return np.array(
        [
            scorer.score(r, p)["rougeL"].fmeasure
            for p, r in zip(predictions, references, strict=True)
        ]
    )


def bertscore(
    predictions: Sequence[str], references: Sequence[str], model_type: str = "roberta-large"
) -> dict[str, np.ndarray]:
    from bert_score import score

    p, r, f = score(
        _nonempty(predictions),
        list(references),
        model_type=model_type,
        batch_size=32,
        verbose=False,
    )
    return {
        "bertscore_precision": p.numpy(),
        "bertscore_recall": r.numpy(),
        "bertscore_f1": f.numpy(),
    }


def semantic_similarity(
    predictions: Sequence[str], references: Sequence[str], model_name: str
) -> np.ndarray:
    """Cosine similarity of biomedical sentence embeddings (replaces the 21 GB BioSentVec)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    a = model.encode(_nonempty(predictions), normalize_embeddings=True, batch_size=64)
    b = model.encode(list(references), normalize_embeddings=True, batch_size=64)
    return (np.asarray(a) * np.asarray(b)).sum(axis=1)


def compute_all(
    predictions: Sequence[str], references: Sequence[str], cfg: EvalConfig
) -> dict[str, np.ndarray]:
    scores = bertscore(predictions, references, cfg.bertscore_model)
    scores["rouge_l"] = rouge_l(predictions, references)
    scores["semantic_cosine"] = semantic_similarity(predictions, references, cfg.semantic_model)
    return scores


def summarize(scores: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {name: bootstrap_ci(values) for name, values in scores.items()}
