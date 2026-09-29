from __future__ import annotations

import torch
from transformers import TrainerCallback


class GpuMemoryCallback(TrainerCallback):
    """Adds peak reserved GPU memory (GB) to every log line so it appears in TensorBoard."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and torch.cuda.is_available():
            logs["gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)


def gpu_summary() -> dict:
    if not torch.cuda.is_available():
        return {"gpu": None}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "gpu_total_gb": round(props.total_memory / 1024**3, 2),
        "gpu_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    }
