"""Release a chosen checkpoint to the Hugging Face Hub as a tagged, documented adapter."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from medqa.config import ExperimentConfig

MODEL_CARD = """---
base_model: {base_model}
library_name: peft
license: apache-2.0
datasets:
- medalpaca/medical_meadow_medical_flashcards
tags: [lora, qlora, unsloth, medical, question-answering]
---

# {display_name} - medical flashcard QA adapter ({tag})

LoRA adapter for `{base_model}`, fine-tuned on Medical Meadow flashcards with the
`medqa` training pipeline. Trained on a single NVIDIA T4.

**Not for clinical use.** Research and educational purposes only.

## Evaluation
{metrics}

## Training
```json
{training}
```
"""


def _metrics_table(run_dir: Path) -> str:
    lines = ["| Setting | n | BERTScore F1 | ROUGE-L | Semantic cosine |", "|---|---|---|---|---|"]
    for path in sorted(run_dir.glob("eval/*/metrics.json")):
        m = json.loads(path.read_text())
        mm = m["metrics"]
        lines.append(
            f"| {m['tag']} | {m['n']} | {mm['bertscore_f1']['mean']:.3f} | {mm['rouge_l']['mean']:.3f} | {mm['semantic_cosine']['mean']:.3f} |"
        )
    return "\n".join(lines) if len(lines) > 2 else "_Not evaluated yet._"


def push_adapter(
    cfg: ExperimentConfig, adapter: Path, repo_id: str, tag: str, private: bool = True
) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    summary_path = cfg.run_dir / "train_summary.json"
    training = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "adapter"
        shutil.copytree(
            adapter,
            staging,
            ignore=shutil.ignore_patterns(
                "optimizer.pt", "scheduler.pt", "rng_state*", "global_step*"
            ),
        )
        (staging / "README.md").write_text(
            MODEL_CARD.format(
                base_model=cfg.serve.base_model,
                display_name=cfg.model.display_name,
                tag=tag,
                metrics=_metrics_table(cfg.run_dir),
                training=json.dumps(training, indent=2),
            )
        )
        commit = api.upload_folder(
            repo_id=repo_id, folder_path=str(staging), commit_message=f"Release {tag}"
        )
    api.create_tag(repo_id, tag=tag, revision=commit.oid, exist_ok=True)
    return f"https://huggingface.co/{repo_id}/tree/{tag}"


def push_splits(split_dir: Path, repo_id: str, private: bool = False) -> str:
    from datasets import Dataset, DatasetDict

    from medqa.data.split import load_splits

    splits = load_splits(split_dir)
    DatasetDict(
        {name: Dataset.from_pandas(df, preserve_index=False) for name, df in splits.items()}
    ).push_to_hub(repo_id, private=private)
    return f"https://huggingface.co/datasets/{repo_id}"
