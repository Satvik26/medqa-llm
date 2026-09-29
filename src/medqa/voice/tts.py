"""Text-to-speech with a small VITS model (replaces the older Tacotron2 + WaveGlow pair)."""

from __future__ import annotations

from functools import cached_property

import numpy as np


class Speaker:
    def __init__(self, model: str = "facebook/mms-tts-eng") -> None:
        self.model = model

    @cached_property
    def pipe(self):
        import torch
        from transformers import pipeline

        return pipeline(
            "text-to-speech", model=self.model, device=0 if torch.cuda.is_available() else -1
        )

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        out = self.pipe(text)
        return int(out["sampling_rate"]), np.squeeze(out["audio"]).astype(np.float32)
