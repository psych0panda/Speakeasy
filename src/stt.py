"""speakeasy.stt

Потоковое распознавание речи на базе faster-whisper.

Основная идея: кормим модель аудио кусочками (чанками) по ~0.5 с с
перекрытием 0.2 с, извлекаем промежуточные (`non-final`) результаты и
отдаём их вызывающему коду через асинхронный генератор.

Пример использования:
    import asyncio, soundfile as sf
    from speakeasy.stt import WhisperStreamer

    async def main():
        model = WhisperStreamer(device="auto")
        audio, sr = sf.read("sample.wav", dtype="float32")
        async for segment in model.transcribe_iter(audio, sr):
            print(segment.text)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import numpy as np
from faster_whisper import WhisperModel, Segment

__all__ = [
    "TranscribedSegment",
    "WhisperStreamer",
]

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TranscribedSegment:
    text: str
    start: float
    end: float
    is_final: bool = True  # False для промежуточных гипотез

    def __str__(self) -> str:
        t = self.text.rstrip()
        stamp = f"{self.start:6.2f}-{self.end:6.2f}"
        return f"[{stamp}] {t}{'…' if not self.is_final else ''}"


# ---------------------------------------------------------------------------
# Core streamer
# ---------------------------------------------------------------------------


class WhisperStreamer:
    """Обёртка над faster-whisper, поддерживающая инкрементальный ввод аудио."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_threshold: float = 0.5,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type or (
            "int8_float16" if device != "cpu" else "int8"
        )
        _LOG.info("Loading faster-whisper model %s on %s", model_size, device)
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=self.compute_type,
            download_root=str(Path.home() / ".cache" / "whisper"),
        )
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.vad_threshold = vad_threshold

        # state
        self._prev_text: str = ""
        self._prev_offset: float = 0.0

    # ------------------------------------------------------------------
    # Public high-level API – async generator
    # ------------------------------------------------------------------

    async def transcribe_iter(
        self,
        audio: np.ndarray,
        samplerate: int,
        chunk_size_ms: int = 500,
        step_ms: int = 300,
    ) -> AsyncGenerator[TranscribedSegment, None]:
        """Асинхронно генерирует сегменты распознавания.

        Parameters
        ----------
        audio : np.ndarray [float32 mono]
        samplerate : int
            Частота исходного сигнала.
        chunk_size_ms : int
            Длина окна, подаваемого в модель.
        step_ms : int
            Сдвиг между окнами (<chunk_size_ms => перекрытие).
        """
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)  # to mono
        assert audio.dtype == np.float32, "audio must be float32"

        chunk = int(chunk_size_ms * samplerate / 1000)
        step = int(step_ms * samplerate / 1000)
        if step <= 0 or chunk <= 0:
            raise ValueError("chunk/step must be > 0")

        i = 0
        while i < len(audio):
            window = audio[i : i + chunk]
            if len(window) < chunk:
                # Zero-pad the last chunk
                window = np.pad(window, (0, chunk - len(window)))

            segments: List[Segment] = list(
                self._model.transcribe_iterator(
                    window,
                    language="en",
                    beam_size=self.beam_size,
                    vad_filter=self.vad_filter,
                    vad_parameters={"threshold": self.vad_threshold},
                    word_timestamps=False,
                )
            )

            # Convert segments
            for seg in segments:
                start = self._prev_offset + seg.start
                end = self._prev_offset + seg.end
                yield TranscribedSegment(seg.text, start, end, is_final=True)

            await asyncio.sleep(0)  # let event loop breathe
            i += step
            self._prev_offset += step / samplerate

    # ------------------------------------------------------------------
    # Simple blocking helper
    # ------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray, samplerate: int) -> str:
        """Синхронно распознаёт полный буфер и возвращает текст."""
        result = self._model.transcribe(audio, language="en")
        return " ".join(seg.text for seg in result[0])
