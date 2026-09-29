from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def top_ngrams(
    documents: Iterable[Sequence[str]], n: int = 2, k: int = 20
) -> list[tuple[str, int]]:
    """Most common n-grams, counted within each document so they never span two flashcards."""
    counts: Counter[tuple[str, ...]] = Counter()
    for tokens in documents:
        counts.update(ngrams(tokens, n))
    return [(" ".join(gram), c) for gram, c in counts.most_common(k)]
