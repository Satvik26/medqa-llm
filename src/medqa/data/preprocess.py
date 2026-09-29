"""Text normalisation used by the analysis and retrieval modules."""

from __future__ import annotations

import re
from functools import cache

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "i'm": "i am",
    "it's": "it is",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "there's": "there is",
    "that's": "that is",
    "what's": "what is",
    "let's": "let us",
}
_CONTRACTION_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)), re.IGNORECASE
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def expand_contractions(text: str) -> str:
    return _CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0).lower()], text)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without punctuation; keeps terms such as `beta-blocker` together."""
    return _TOKEN_RE.findall(text.lower())


@cache
def english_stopwords() -> frozenset[str]:
    import nltk
    from nltk.corpus import stopwords

    try:
        return frozenset(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return frozenset(stopwords.words("english"))


@cache
def _lemmatizer():
    import nltk
    from nltk.stem import WordNetLemmatizer

    for resource in ("wordnet", "omw-1.4"):
        nltk.download(resource, quiet=True)
    return WordNetLemmatizer()


def preprocess(
    text: str,
    *,
    expand: bool = True,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
    stop_words: frozenset[str] | None = None,
) -> list[str]:
    if expand:
        text = expand_contractions(text)
    tokens = tokenize(text)
    if remove_stopwords:
        sw = stop_words if stop_words is not None else english_stopwords()
        tokens = [t for t in tokens if t not in sw]
    if lemmatize:
        lem = _lemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]
    return tokens
