import pandas as pd
import pytest

from medqa.data.load import clean_flashcards
from medqa.data.preprocess import expand_contractions, preprocess, tokenize
from medqa.data.split import load_splits, make_splits, overlap_stats, save_splits


def test_clean_flashcards_drops_empty_and_duplicates():
    raw = pd.DataFrame(
        {
            "instruction": ["x"] * 4,
            "input": ["Q1", "Q1", " ", "Q2"],
            "output": ["A1", "A1", "A", "A2 "],
        }
    )
    out = clean_flashcards(raw)
    assert out["question"].tolist() == ["Q1", "Q2"]
    assert out["answer"].tolist() == ["A1", "A2"]
    assert out["id"].tolist() == [0, 1]


def test_splits_are_disjoint_deterministic_and_sized(flashcards):
    a = make_splits(flashcards, test_size=0.2, val_size=0.05, seed=1)
    b = make_splits(flashcards, test_size=0.2, val_size=0.05, seed=1)
    assert {k: len(v) for k, v in a.items()} == {"test": 40, "val": 10, "train": 150}
    ids = [set(a[s]["id"]) for s in ("train", "val", "test")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert all(a[s]["id"].tolist() == b[s]["id"].tolist() for s in a)


def test_split_roundtrip(tmp_path, flashcards):
    splits = make_splits(flashcards)
    save_splits(splits, tmp_path)
    loaded = load_splits(tmp_path)
    pd.testing.assert_frame_equal(loaded["test"], splits["test"])


def test_missing_splits_raise(tmp_path):
    with pytest.raises(FileNotFoundError, match="medqa data split"):
        load_splits(tmp_path)


def test_overlap_stats_detects_duplicates():
    train = pd.DataFrame({"question": ["q1", "q2"], "answer": ["a1", "a2"]})
    test = pd.DataFrame({"question": ["Q1 ", "q3"], "answer": ["a9", "A2"]})
    assert overlap_stats({"train": train, "test": test}) == {
        "test_question_in_train": 0.5,
        "test_answer_in_train": 0.5,
    }


def test_tokenize_keeps_medical_hyphenation():
    assert tokenize("Beta-blockers reduce HR, e.g. (propranolol)!") == [
        "beta-blockers", "reduce", "hr", "e", "g", "propranolol",
    ]  # fmt: skip


def test_contractions_and_stopwords():
    assert expand_contractions("It's not what it isn't") == "it is not what it is not"
    tokens = preprocess(
        "The heart doesn't pump",
        remove_stopwords=True,
        stop_words=frozenset({"the", "does", "not"}),
    )
    assert tokens == ["heart", "pump"]
