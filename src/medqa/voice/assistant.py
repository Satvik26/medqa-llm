"""Voice question answering: speech -> Whisper -> fine-tuned LLM -> speech."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from medqa.voice.stt import Transcriber
from medqa.voice.tts import Speaker


class VoiceAssistant:
    def __init__(
        self,
        answer: Callable[[str], str],
        transcriber: Transcriber | None = None,
        speaker: Speaker | None = None,
    ) -> None:
        self.answer = answer
        self.transcriber = transcriber or Transcriber()
        self.speaker = speaker or Speaker()

    def respond(
        self, audio: str | tuple[int, np.ndarray]
    ) -> tuple[str, str, tuple[int, np.ndarray]]:
        question = self.transcriber.transcribe(audio)
        reply = self.answer(question)
        return question, reply, self.speaker.synthesize(reply)
