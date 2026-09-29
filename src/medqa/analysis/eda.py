"""Exploratory data analysis: sizes, lengths, vocabulary, frequent words and n-grams."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from medqa.analysis import plots
from medqa.analysis.ngrams import top_ngrams
from medqa.data.preprocess import english_stopwords, preprocess


def describe(df: pd.DataFrame) -> dict:
    q_tokens = df["question"].map(preprocess)
    a_tokens = df["answer"].map(preprocess)
    q_len, a_len = q_tokens.map(len), a_tokens.map(len)
    return {
        "pairs": len(df),
        "question_words": {"mean": q_len.mean(), "median": q_len.median(), "max": int(q_len.max())},
        "answer_words": {"mean": a_len.mean(), "median": a_len.median(), "max": int(a_len.max())},
        "question_vocabulary": len({t for toks in q_tokens for t in toks}),
        "answer_vocabulary": len({t for toks in a_tokens for t in toks}),
        "unique_questions": int(df["question"].nunique()),
    }


def run_eda(df: pd.DataFrame, out_dir: Path) -> dict:
    sw = english_stopwords()
    q_tokens = df["question"].map(preprocess)
    a_tokens = df["answer"].map(preprocess)
    q_content = q_tokens.map(lambda ts: [t for t in ts if t not in sw])
    a_content = a_tokens.map(lambda ts: [t for t in ts if t not in sw])

    plots.histogram(
        q_tokens.map(len), "Words per question", "words", out_dir / "question_lengths.png"
    )
    plots.histogram(a_tokens.map(len), "Words per answer", "words", out_dir / "answer_lengths.png")

    for name, docs in (("questions", q_content), ("answers", a_content)):
        freq = Counter(t for toks in docs for t in toks)
        top = freq.most_common(25)
        plots.barh(
            [w for w, _ in top],
            [c for _, c in top],
            f"Top words in {name} (no stopwords)",
            "count",
            out_dir / f"top_words_{name}.png",
        )
        plots.wordcloud(
            dict(freq.most_common(300)), f"Word cloud: {name}", out_dir / f"wordcloud_{name}.png"
        )

    ngram_tables = {}
    for n in (2, 3):
        top = top_ngrams(a_content, n=n, k=20)
        ngram_tables[f"answers_{n}gram"] = top
        plots.barh(
            [g for g, _ in top],
            [c for _, c in top],
            f"Top {n}-grams in answers",
            "count",
            out_dir / f"answers_{n}grams.png",
        )

    stats = describe(df) | {"top_ngrams": ngram_tables}
    (out_dir / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=float))
    return stats
