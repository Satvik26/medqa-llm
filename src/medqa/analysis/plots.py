"""Matplotlib helpers that write figures to disk (headless-safe, consistent style)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#4b5563"]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    }
)


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def histogram(values: Sequence[float], title: str, xlabel: str, path: Path, bins: int = 50) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=bins, color=PALETTE[0], alpha=0.85)
    ax.set(title=title, xlabel=xlabel, ylabel="Count")
    return save(fig, path)


def barh(
    labels: Sequence[str], values: Sequence[float], title: str, xlabel: str, path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(labels))))
    ax.barh(list(labels)[::-1], list(values)[::-1], color=PALETTE[0])
    ax.set(title=title, xlabel=xlabel)
    return save(fig, path)


def lines(
    x: Sequence[float],
    series: Mapping[str, Sequence[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    mark_best: str | None = None,
) -> Path:
    """Plot one line per series; `mark_best` names a series whose maximum gets annotated."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, ys in series.items():
        ax.plot(x, ys, marker="o", markersize=3, label=label)
    if mark_best and mark_best in series:
        ys = list(series[mark_best])
        best = max(range(len(ys)), key=ys.__getitem__)
        ax.axvline(x[best], color="grey", linestyle="--", linewidth=1)
        ax.annotate(
            f"best: {x[best]}", (x[best], ys[best]), xytext=(5, -15), textcoords="offset points"
        )
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if len(series) > 1:
        ax.legend()
    return save(fig, path)


def wordcloud(frequencies: Mapping[str, int], title: str, path: Path) -> Path:
    from wordcloud import WordCloud

    cloud = WordCloud(width=1200, height=600, background_color="white", colormap="viridis")
    cloud.generate_from_frequencies(dict(frequencies))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.set_axis_off()
    ax.set_title(title)
    return save(fig, path)
