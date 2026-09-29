import json

import pytest

from medqa.training.checkpoints import (
    best_checkpoint,
    eval_losses,
    latest_checkpoint,
    list_checkpoints,
    resolve_resume,
)


@pytest.fixture
def run_dir(tmp_path):
    for step in (100, 300, 200):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "adapter").mkdir()
    state = {
        "best_model_checkpoint": str(tmp_path / "checkpoint-200"),
        "log_history": [
            {"step": 100, "eval_loss": 1.1},
            {"step": 200, "eval_loss": 0.9},
            {"step": 300, "eval_loss": 0.95},
        ],
    }
    (tmp_path / "checkpoint-300" / "trainer_state.json").write_text(json.dumps(state))
    return tmp_path


def test_checkpoints_sorted_numerically(run_dir):
    assert [c.step for c in list_checkpoints(run_dir)] == [100, 200, 300]
    assert latest_checkpoint(run_dir).step == 300


def test_best_checkpoint_from_newest_state(run_dir):
    assert best_checkpoint(run_dir).name == "checkpoint-200"


def test_eval_losses():
    assert eval_losses(
        {"log_history": [{"step": 5, "loss": 2.0}, {"step": 5, "eval_loss": 1.5}]}
    ) == {5: 1.5}


def test_resolve_resume(run_dir, tmp_path_factory):
    assert resolve_resume(None, run_dir, None) is None
    assert resolve_resume("auto", run_dir, None).endswith("checkpoint-300")
    assert resolve_resume("auto", tmp_path_factory.mktemp("empty"), None) is None
    assert resolve_resume(str(run_dir / "checkpoint-100"), run_dir, None).endswith("checkpoint-100")
    with pytest.raises(ValueError):
        resolve_resume("hub", run_dir, None)
    with pytest.raises(FileNotFoundError):
        resolve_resume("does/not/exist", run_dir, None)
