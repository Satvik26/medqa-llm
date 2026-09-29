"""Generate answers for an evaluation split, score them, and write predictions + metrics to disk.

Layout: runs/<run_name>/eval/<tag>/{predictions.jsonl, metrics.json}
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rich import print

from medqa.config import ApiConfig, ExperimentConfig
from medqa.data.split import load_splits
from medqa.evaluation.metrics import bertscore, compute_all, rouge_l, summarize
from medqa.inference.base import GenerationBatch
from medqa.training.checkpoints import eval_losses, list_checkpoints, read_trainer_state

Backend = Literal["vllm", "api", "hf"]


def eval_frame(cfg: ExperimentConfig, split: str, limit: int | None, rag: bool) -> pd.DataFrame:
    split_dir = (
        cfg.data.split_dir.with_name(cfg.data.split_dir.name + "_rag")
        if rag
        else cfg.data.split_dir
    )
    frame = load_splits(split_dir)[split]
    if limit and limit < len(frame):
        frame = frame.sample(n=limit, random_state=cfg.data.seed).sort_values("id")
    return frame.reset_index(drop=True)


def default_tag(adapter: Path | None, rag: bool) -> str:
    return ("finetuned" if adapter else "zero-shot") + ("+rag" if rag else "")


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _generate(
    cfg: ExperimentConfig, backend: Backend, adapter: Path | None, frame: pd.DataFrame, rag: bool
) -> GenerationBatch:
    contexts = [list(c) for c in frame["contexts"]] if rag else None
    questions = frame["question"].tolist()
    if backend == "vllm":
        from medqa.inference.vllm_engine import VLLMGenerator

        gen = VLLMGenerator(cfg, enable_lora=adapter is not None).use_adapter(adapter)
    elif backend == "api":
        from medqa.inference.openai_engine import OpenAIGenerator

        api = cfg.api or ApiConfig(
            base_url=f"http://localhost:{cfg.serve.port}/v1",
            model=cfg.serve.served_lora_name if adapter else cfg.serve.base_model,
        )
        gen = OpenAIGenerator(api, cfg.generation)
    else:
        from medqa.inference.hf_engine import HFGenerator

        gen = HFGenerator(
            cfg.serve.base_model, adapter, max_new_tokens=cfg.generation.max_new_tokens
        )
    return gen.generate(questions, contexts)


def _write_jsonl(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(path, orient="records", lines=True, force_ascii=False)


def evaluate(
    cfg: ExperimentConfig,
    backend: Backend = "vllm",
    adapter: Path | None = None,
    rag: bool = False,
    split: str | None = None,
    limit: int | None = None,
    tag: str | None = None,
    judge_api: ApiConfig | None = None,
    judge_sample: int = 0,
) -> dict:
    split = split or cfg.eval.split
    limit = limit if limit is not None else cfg.eval.limit
    tag = tag or default_tag(adapter, rag)
    out_dir = cfg.run_dir / "eval" / tag
    frame = eval_frame(cfg, split, limit, rag)
    print(
        f"[bold]Evaluating[/bold] {cfg.model.display_name} [{tag}] on {len(frame)} {split} questions via {backend}"
    )

    batch = _generate(cfg, backend, adapter, frame, rag)
    _free_gpu()

    preds = frame[["id", "question", "answer"]].rename(columns={"answer": "reference"})
    preds["prediction"] = batch.texts
    preds["completion_tokens"] = batch.completion_tokens
    scores = compute_all(preds["prediction"].tolist(), preds["reference"].tolist(), cfg.eval)
    for name, values in scores.items():
        preds[name] = np.round(values, 4)
    if rag:
        preds["answer_in_context"] = frame["answer_in_context"].to_numpy()

    metrics = {
        "run_name": cfg.run_name,
        "display_name": cfg.model.display_name,
        "tag": tag,
        "backend": backend,
        "model": cfg.api.model if (backend == "api" and cfg.api) else cfg.serve.base_model,
        "adapter": str(adapter) if adapter else None,
        "quantization": cfg.serve.quantization,
        "rag": rag,
        "split": split,
        "n": len(preds),
        "metrics": summarize(scores),
        "throughput": batch.throughput,
    }
    if judge_api and judge_sample:
        from medqa.evaluation.judge import judge

        sample = preds.sample(n=min(judge_sample, len(preds)), random_state=0)
        judged = judge(
            judge_api,
            sample["question"].tolist(),
            sample["prediction"].tolist(),
            sample["reference"].tolist(),
        )
        valid = [s for s in judged if s is not None]
        metrics["judge"] = {
            "model": judge_api.model,
            "n": len(valid),
            **summarize({"score": np.array(valid, dtype=float)})["score"],
        }

    _write_jsonl(preds, out_dir / "predictions.jsonl")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print({k: round(v["mean"], 4) for k, v in metrics["metrics"].items()}, metrics["throughput"])
    return metrics


def rescore(predictions_path: Path, cfg: ExperimentConfig) -> dict:
    """Recompute metrics for an existing predictions file (e.g. after changing a metric)."""
    preds = pd.read_json(predictions_path, lines=True)
    scores = compute_all(preds["prediction"].tolist(), preds["reference"].tolist(), cfg.eval)
    metrics_path = predictions_path.with_name("metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["metrics"] = summarize(scores)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def evaluate_checkpoints(cfg: ExperimentConfig, sample: int | None = None) -> list[dict]:
    """Score every saved checkpoint on a fixed validation sample with a single vLLM engine."""
    from medqa.inference.vllm_engine import VLLMGenerator

    ckpts = list_checkpoints(cfg.run_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* directories in {cfg.run_dir}")
    frame = eval_frame(cfg, "val", sample or cfg.eval.checkpoint_sample, cfg.data.rag)
    contexts = [list(c) for c in frame["contexts"]] if cfg.data.rag else None
    gen = VLLMGenerator(cfg, enable_lora=True)
    outputs = {
        c.step: gen.use_adapter(c.path).generate(frame["question"].tolist(), contexts).texts
        for c in ckpts
    }
    del gen
    _free_gpu()

    state = read_trainer_state(cfg.run_dir) or {}
    losses = eval_losses(state)
    refs = frame["answer"].tolist()
    rows = []
    for step, texts in outputs.items():
        f1 = bertscore(texts, refs, cfg.eval.bertscore_model)["bertscore_f1"]
        rows.append(
            {
                "step": step,
                "bertscore_f1": float(f1.mean()),
                "rouge_l": float(rouge_l(texts, refs).mean()),
                "eval_loss": losses.get(step),
            }
        )
    best = max(rows, key=lambda r: r["bertscore_f1"])
    result = {"sample": len(frame), "best_step": best["step"], "checkpoints": rows}
    (cfg.run_dir / "checkpoint_scores.json").write_text(json.dumps(result, indent=2))
    print(result)
    return rows
