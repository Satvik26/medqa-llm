"""TF-IDF + k-means topic clustering of flashcards, with k chosen by silhouette score."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from medqa.analysis import plots


def vectorize(texts: Sequence[str], max_df: float = 0.8, min_df: int = 5):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words="english")
    return vectorizer.fit_transform(texts), vectorizer


def scan_k(X, k_values: Sequence[int], seed: int = 2307, sample_size: int = 5000) -> pd.DataFrame:
    """Inertia and (sampled) silhouette for each k, using MiniBatchKMeans for speed."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    rows = []
    for k in k_values:
        km = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init=3, random_state=seed).fit(X)
        sil = silhouette_score(
            X, km.labels_, sample_size=min(sample_size, X.shape[0]), random_state=seed
        )
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def top_terms(centroids: np.ndarray, vocabulary: np.ndarray, n: int = 10) -> list[list[str]]:
    return [list(vocabulary[np.argsort(c)[::-1][:n]]) for c in centroids]


def run_clustering(
    texts: Sequence[str],
    out_dir: Path,
    k: int | None = None,
    k_values: Sequence[int] = tuple(range(5, 55, 5)),
    seed: int = 2307,
) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import silhouette_score

    X, vectorizer = vectorize(texts)
    scan = scan_k(X, k_values, seed=seed)
    scan.to_csv(out_dir / "k_scan.csv", index=False)
    plots.lines(
        scan["k"].tolist(),
        {"silhouette": scan["silhouette"].tolist()},
        "Choosing k: silhouette score",
        "k",
        "silhouette",
        out_dir / "k_silhouette.png",
        mark_best="silhouette",
    )
    plots.lines(
        scan["k"].tolist(),
        {"inertia": scan["inertia"].tolist()},
        "Choosing k: elbow (inertia)",
        "k",
        "within-cluster SSE",
        out_dir / "k_elbow.png",
    )

    k = k or int(scan.loc[scan["silhouette"].idxmax(), "k"])
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X)
    terms = top_terms(km.cluster_centers_, vectorizer.get_feature_names_out())
    sizes = np.bincount(km.labels_, minlength=k)
    labels = [f"C{i}: {', '.join(t[:3])}" for i, t in enumerate(terms)]
    plots.barh(
        labels,
        sizes.tolist(),
        f"Cluster sizes (k={k})",
        "flashcards",
        out_dir / "cluster_sizes.png",
    )

    coords = TruncatedSVD(n_components=2, random_state=seed).fit_transform(X)
    fig, ax = plots.plt.subplots(figsize=(8, 6))
    idx = np.random.default_rng(seed).choice(X.shape[0], size=min(5000, X.shape[0]), replace=False)
    ax.scatter(coords[idx, 0], coords[idx, 1], c=km.labels_[idx], cmap="tab20", s=4, alpha=0.6)
    ax.set(
        title=f"Flashcards in 2-D (TruncatedSVD), coloured by cluster (k={k})",
        xlabel="SVD 1",
        ylabel="SVD 2",
    )
    plots.save(fig, out_dir / "clusters_svd.png")

    summary = {
        "k": k,
        "silhouette": float(
            silhouette_score(X, km.labels_, sample_size=min(5000, X.shape[0]), random_state=seed)
        ),
        "inertia": float(km.inertia_),
        "clusters": [
            {"id": i, "size": int(s), "top_terms": t}
            for i, (s, t) in enumerate(zip(sizes, terms, strict=True))
        ],
    }
    (out_dir / "clusters.json").write_text(json.dumps(summary, indent=2))
    return summary
