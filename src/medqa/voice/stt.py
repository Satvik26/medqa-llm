"""Speech-to-text with Whisper through the transformers pipeline."""

from __future__ import annotations

from functools import cached_property

import numpy as np


class Transcriber:
    def __init__(self, model: str = "openai/whisper-small") -> None:
        self.model = model

    @cached_property
    def pipe(self):
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        return pipeline(
            "automatic-speech-recognition", model=self.model, device=device, dtype=dtype
        )

    def transcribe(self, audio: str | tuple[int, np.ndarray]) -> str:
        """`audio` is a file path or a `(sample_rate, samples)` tuple as produced by Gradio."""
        if isinstance(audio, tuple):
            sr, samples = audio
            samples = samples.astype(np.float32)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            if np.abs(samples).max() > 1:
                samples /= 32768.0
            audio = {"sampling_rate": sr, "raw": samples}
        result = self.pipe(audio, generate_kwargs={"language": "english"}, chunk_length_s=30)
        return result["text"].strip()
