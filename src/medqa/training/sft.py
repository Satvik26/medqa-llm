"""QLoRA / LoRA supervised fine-tuning with Unsloth + TRL, with resumable checkpoints."""

from __future__ import annotations

import dataclasses
import inspect
import json
import time
from typing import Any

from rich import print

from medqa.config import ExperimentConfig
from medqa.data.split import load_splits
from medqa.prompts import render_example
from medqa.training.checkpoints import resolve_resume


def _accepted(target: Any) -> set[str]:
    if dataclasses.is_dataclass(target):
        return {f.name for f in dataclasses.fields(target)}
    return set(inspect.signature(target).parameters)


def _pick(kwargs: dict[str, Any], accepted: set[str]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in accepted}


def _to_dataset(frame, tokenizer, use_contexts: bool):
    from datasets import Dataset

    contexts = frame["contexts"] if use_contexts else [None] * len(frame)
    texts = [
        render_example(tokenizer, q, a, list(c) if c is not None else None)
        for q, a, c in zip(frame["question"], frame["answer"], contexts, strict=True)
    ]
    return Dataset.from_dict({"text": texts})


def train(cfg: ExperimentConfig, resume: str | None = None) -> dict:
    # Unsloth must be imported before transformers/trl so its patches apply.
    from unsloth import FastLanguageModel, is_bfloat16_supported  # isort: skip
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    from medqa.training.callbacks import GpuMemoryCallback, gpu_summary

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(run_dir / "config.yaml")
    resume_from = resolve_resume(resume, run_dir, cfg.training.hub_model_id)
    print(f"[bold]Run[/bold] {run_dir}  resume_from={resume_from}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.base_model,
        max_seq_length=cfg.model.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    lora = cfg.model.lora
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=lora.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.training.seed,
    )

    splits = load_splits(cfg.data.active_split_dir)
    train_ds = _to_dataset(splits["train"], tokenizer, cfg.data.rag)
    val_ds = _to_dataset(splits["val"], tokenizer, cfg.data.rag)

    t = cfg.training
    bf16 = is_bfloat16_supported()
    # Both spellings are offered because TRL / transformers renamed these arguments across releases.
    args_kwargs = {
        "output_dir": str(run_dir),
        "dataset_text_field": "text",
        "max_seq_length": cfg.model.max_seq_length,
        "max_length": cfg.model.max_seq_length,
        "packing": False,
        "dataset_num_proc": 2,
        "per_device_train_batch_size": t.per_device_train_batch_size,
        "per_device_eval_batch_size": t.per_device_train_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "num_train_epochs": t.num_train_epochs,
        "max_steps": t.max_steps,
        "learning_rate": t.learning_rate,
        "warmup_ratio": t.warmup_ratio,
        "weight_decay": t.weight_decay,
        "lr_scheduler_type": t.lr_scheduler_type,
        "optim": t.optim,
        "fp16": not bf16,
        "bf16": bf16,
        "logging_steps": t.logging_steps,
        "logging_dir": str(run_dir / "logs"),
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps": t.eval_steps,
        "save_strategy": "steps",
        "save_steps": t.save_steps,
        "save_total_limit": t.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": t.report_to,
        "seed": t.seed,
        "push_to_hub": t.push_to_hub,
        "hub_model_id": t.hub_model_id,
        "hub_strategy": "checkpoint",
        "hub_private_repo": True,
    }
    args = SFTConfig(**_pick(args_kwargs, _accepted(SFTConfig)))

    callbacks = [GpuMemoryCallback()]
    if t.early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=t.early_stopping_patience))
    trainer_params = _accepted(SFTTrainer.__init__)
    tokenizer_arg = "processing_class" if "processing_class" in trainer_params else "tokenizer"
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=callbacks,
        **{tokenizer_arg: tokenizer},
    )

    start = time.time()
    result = trainer.train(resume_from_checkpoint=resume_from)
    trainer.state.save_to_json(str(run_dir / "trainer_state.json"))

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "run_name": cfg.run_name,
        "display_name": cfg.model.display_name,
        "base_model": cfg.model.base_model,
        "load_in_4bit": cfg.model.load_in_4bit,
        "rag": cfg.data.rag,
        "train_rows": len(train_ds),
        "global_step": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "train_loss": result.training_loss,
        "wall_time_s": round(time.time() - start, 1),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        **gpu_summary(),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)
    return summary
