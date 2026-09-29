"""Domain Word2Vec embeddings trained on the flashcards, compared against a general-purpose model."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from medqa.analysis import plots

PROBE_TERMS = ("insulin", "heart", "virus", "pregnancy", "antidepressant", "kidney", "tumor")


def train_word2vec(
    sentences: Sequence[Sequence[str]],
    vector_size: int = 100,
    window: int = 10,
    min_count: int = 5,
    seed: int = 42,
):
    from gensim.models import Word2Vec

    return Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=seed,
        epochs=10,
    )


def neighbours(
    keyed_vectors, terms: Sequence[str], topn: int = 8
) -> dict[str, list[tuple[str, float]]]:
    return {
        t: [(w, round(float(s), 3)) for w, s in keyed_vectors.most_similar(t, topn=topn)]
        for t in terms
        if t in keyed_vectors
    }


def plot_tsne(keyed_vectors, words: Sequence[str], path: Path, seed: int = 42) -> Path:
    from sklearn.manifold import TSNE

    words = [w for w in words if w in keyed_vectors]
    coords = TSNE(
        n_components=2, perplexity=min(30, len(words) - 1), random_state=seed
    ).fit_transform(np.stack([keyed_vectors[w] for w in words]))
    fig, ax = plots.plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.5)
    for (x, y), w in list(zip(coords, words, strict=True))[:120]:
        ax.annotate(w, (x, y), fontsize=7, alpha=0.8)
    ax.set(title="t-SNE of domain Word2Vec embeddings", xticks=[], yticks=[])
    return plots.save(fig, path)


def run_embeddings(
    sentences: Sequence[Sequence[str]], out_dir: Path, pretrained: str | None = None
) -> dict:
    """Train on the flashcards; optionally compare neighbours with a gensim-downloader model."""
    model = train_word2vec(sentences)
    kv = model.wv
    result = {"vocabulary_size": len(kv), "domain_neighbours": neighbours(kv, PROBE_TERMS)}
    frequent = kv.index_to_key[50:450]
    plot_tsne(kv, [*PROBE_TERMS, *frequent], out_dir / "word2vec_tsne.png")

    if pretrained:
        import gensim.downloader as api

        result["pretrained_model"] = pretrained
        result["pretrained_neighbours"] = neighbours(api.load(pretrained), PROBE_TERMS)

    (out_dir / "embeddings.json").write_text(json.dumps(result, indent=2))
    return result
