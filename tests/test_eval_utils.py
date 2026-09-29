import json

import numpy as np

from medqa.analysis.ngrams import ngrams, top_ngrams
from medqa.evaluation.judge import build_judge_prompt, parse_score
from medqa.evaluation.metrics import bootstrap_ci
from medqa.evaluation.report import collect_results, plot_loss_curves, results_markdown
from medqa.inference.base import GenerationBatch
from medqa.retrieval.hybrid import reciprocal_rank_fusion


def test_bootstrap_ci_brackets_mean():
    values = np.random.default_rng(0).normal(0.8, 0.05, size=500)
    ci = bootstrap_ci(values)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert ci["hi"] - ci["lo"] < 0.02


def test_bootstrap_ci_constant_values():
    assert bootstrap_ci([0.5] * 10) == {"mean": 0.5, "lo": 0.5, "hi": 0.5}


def test_parse_judge_score():
    assert parse_score("Feedback: good. [RESULT] 4") == 4
    assert parse_score("Feedback: [RESULT] (5)") == 5
    assert parse_score("no score here") is None
    assert "Reference Answer (Score 5):\nref" in build_judge_prompt("q", "r", "ref")


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert [doc for doc, _ in fused[:2]] in ([1, 2], [2, 1])
    assert fused[-1][0] in (3, 4)
    assert dict(fused)[1] == dict(fused)[2]


def test_ngrams_do_not_cross_documents():
    assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    assert top_ngrams([["a", "b"], ["b", "a"]], n=2, k=5) == [("a b", 1), ("b a", 1)]


def test_throughput_stats():
    batch = GenerationBatch(
        texts=["x", "y"], completion_tokens=[10, 30], seconds=2.0, latencies=[0.5, 1.5]
    )
    assert batch.throughput == {
        "seconds": 2.0, "questions_per_s": 1.0, "tokens_per_s": 20.0, "mean_completion_tokens": 20.0, "mean_latency_s": 1.0,
    }  # fmt: skip


def _metrics(tag, f1):
    ci = {"mean": f1, "lo": f1 - 0.01, "hi": f1 + 0.01}
    return {
        "display_name": "Qwen3-1.7B", "run_name": "qwen", "tag": tag, "n": 100, "quantization": None, "rag": False,
        "metrics": {"bertscore_f1": ci, "rouge_l": ci, "semantic_cosine": ci},
        "throughput": {"questions_per_s": 12.5},
    }  # fmt: skip


def test_report_collects_and_renders(tmp_path):
    for tag, f1 in (("zero-shot", 0.82), ("finetuned", 0.91)):
        d = tmp_path / "qwen" / "eval" / tag
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps(_metrics(tag, f1)))
    df = collect_results(tmp_path)
    assert set(df["tag"]) == {"zero-shot", "finetuned"}
    table = results_markdown(df).splitlines()
    assert "finetuned" in table[2] and "0.910 [0.900, 0.920]" in table[2]


def test_loss_curve_plot(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    state = {
        "best_model_checkpoint": str(run / "checkpoint-20"),
        "log_history": [
            {"step": 10, "loss": 1.2},
            {"step": 20, "loss": 0.9},
            {"step": 20, "eval_loss": 1.0},
        ],
    }
    (run / "trainer_state.json").write_text(json.dumps(state))
    out = plot_loss_curves(run, tmp_path / "loss.png")
    assert out is not None and out.stat().st_size > 0
