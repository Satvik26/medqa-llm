"""Find, rank and restore Hugging Face Trainer checkpoints (no GPU libraries needed)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: Path


def list_checkpoints(run_dir: Path) -> list[Checkpoint]:
    if not run_dir.exists():
        return []
    found = [
        Checkpoint(int(m.group(1)), p)
        for p in run_dir.iterdir()
        if p.is_dir() and (m := _CKPT_RE.match(p.name))
    ]
    return sorted(found, key=lambda c: c.step)


def latest_checkpoint(run_dir: Path) -> Checkpoint | None:
    ckpts = list_checkpoints(run_dir)
    return ckpts[-1] if ckpts else None


def read_trainer_state(run_dir: Path) -> dict | None:
    """The run-level state if training finished, else the one inside the newest checkpoint."""
    candidates = [run_dir / "trainer_state.json"]
    if (latest := latest_checkpoint(run_dir)) is not None:
        candidates.append(latest.path / "trainer_state.json")
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    return None


def eval_losses(state: dict) -> dict[int, float]:
    return {e["step"]: e["eval_loss"] for e in state.get("log_history", []) if "eval_loss" in e}


def train_losses(state: dict) -> dict[int, float]:
    return {e["step"]: e["loss"] for e in state.get("log_history", []) if "loss" in e}


def best_checkpoint(run_dir: Path) -> Path | None:
    state = read_trainer_state(run_dir)
    best = state.get("best_model_checkpoint") if state else None
    return Path(best) if best else None


def download_last_checkpoint(repo_id: str, run_dir: Path) -> Path:
    """Fetch `last-checkpoint/` that `hub_strategy="checkpoint"` keeps on the Hub during training."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, allow_patterns=["last-checkpoint/*"], local_dir=run_dir)
    return run_dir / "last-checkpoint"


def resolve_resume(resume: str | None, run_dir: Path, hub_model_id: str | None) -> str | None:
    """`None` = fresh run, `auto` = newest local checkpoint, `hub` = Hub backup, otherwise a path."""
    if resume in (None, "", "no"):
        return None
    if resume == "auto":
        latest = latest_checkpoint(run_dir)
        return str(latest.path) if latest else None
    if resume == "hub":
        if not hub_model_id:
            raise ValueError("--resume hub needs training.hub_model_id in the config")
        return str(download_last_checkpoint(hub_model_id, run_dir))
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)
