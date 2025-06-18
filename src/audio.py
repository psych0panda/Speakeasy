"""speakeasy.audio

Захват системного звука (WASAPI loopback) либо другого выбранного input-устройства.
По умолчанию ищем loopback-устройство (через PortAudio / WASAPI). Если не найдено,
можно указать индекс вручную опцией --device, например для «Stereo Mix».
"""

from __future__ import annotations

import argparse
import queue
import sys
import wave
from typing import Generator, Optional

import numpy as np
import sounddevice as sd

__all__ = [
    "list_devices_verbose",
    "find_default_loopback_device",
    "LoopbackRecorder",
]

DEFAULT_SAMPLE_RATE = 48000  # Гц
DEFAULT_CHANNELS = 2  # will be resolved per device if None
DTYPE = "int16"  # оптимально для большинства STT движков
DEVICE_INDEX = None  # Если указано, то будет использоваться это устройство по умолчанию
DURATION = 5  # seconds, for testing


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def list_devices_verbose() -> None:
    """Выводит таблицу со всеми аудио-устройствами."""
    for idx, dev in enumerate(sd.query_devices()):
        ha = sd.query_hostapis(dev["hostapi"])["name"]
        print(
            f"[{idx:2}] {dev['name']} | in {dev['max_input_channels']} "
            f"out {dev['max_output_channels']} | {ha}"
        )


def find_default_loopback_device() -> Optional[int]:
    """Ищем устройство для системного loopback-захвата.

    Таким устройством является "CABLE Output (VB-Audio Virtual Cable)"
    поскольку оно регистрируется как output-устройство, но при этом
    поддерживает loopback-захват.
    """

    for idx, dev in enumerate(sd.query_devices()):
        ha = sd.query_hostapis(dev["hostapi"])["name"]
        if dev["name"] == "CABLE Output (VB-Audio Virtual Cable)":
            if dev["max_input_channels"] == 2 and dev["max_output_channels"] == 0:
                if ha == "Windows WASAPI":
                    print(f"Выбранное устройство: {dev['name']}")
                    return dev["index"]

    return None


# ---------------------------------------------------------------------------
# Recorder class
# ---------------------------------------------------------------------------


class LoopbackRecorder:
    """Потоковый захват аудио.

    Если передан `device`, используется он; иначе пытаемся найти loopback.
    """

    def __init__(
        self,
        device: int | None = None,
        samplerate: int = DEFAULT_SAMPLE_RATE,
        blocksize: int = 0,
        channels: int | None = DEFAULT_CHANNELS,
    ) -> None:
        if sys.platform != "win32":
            raise RuntimeError("LoopbackRecorder поддерживает только Windows")

        if device is None:
            device = find_default_loopback_device()
            if device is None:
                raise RuntimeError("WASAPI loopback устройство не найдено")

        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        devinfo = sd.query_devices(device)

        self.channels = channels
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=20)

        # Реальный PortAudio stream создаём лениво в .start(),
        # чтобы не занимать устройство зря, если объект только конфигурируется.
        self._stream: sd.InputStream | None = None

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------
    def __enter__(self) -> "LoopbackRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _create_stream(self) -> sd.InputStream:
        """Вспомогательный метод: создаёт и возвращает InputStream."""
        kwargs = dict(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype=DTYPE,
            channels=self.channels,
            device=self.device,
            latency="low",
            callback=self._callback,
        )
        return sd.InputStream(**kwargs)

    def start(self) -> None:
        """Создаёт (при необходимости) и запускает stream."""
        if self._stream is None:
            self._stream = self._create_stream()
        if not self._stream.active:
            self._stream.start()

    def stop(self) -> None:
        """Останавливает и закрывает stream (если был создан)."""
        if self._stream is None:
            return
        if self._stream.active:
            self._stream.stop()
        # Закрываем устройство, чтобы освободить его для других приложений
        self._stream.close()
        self._stream = None

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Синхронный генератор PCM-фреймов (dtype=int16)."""
        while True:
            data = self._q.get()
            yield data

    # ------------------------------------------------------------------
    # Internal callback
    # ------------------------------------------------------------------
    def _callback(self, indata, frames, time, status):  # noqa: N802
        if status:
            print(f"SoundDevice status: {status}", file=sys.stderr)
        self._q.put(indata.copy())


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------


def _record_cli(seconds: int, outfile: str | None, device_idx: Optional[int]) -> None:
    if device_idx is None:
        print(
            "Не найдено loopback-устройство. Укажите --device IDX, "
            "где IDX — индекс из --list",
            file=sys.stderr,
        )
        sys.exit(1)
    with LoopbackRecorder(device=device_idx) as rec, wave.open(outfile, "wb") as wav:
        # задаём параметры выходного WAV-файла
        wav.setnchannels(rec.channels)
        wav.setsampwidth(2)  # int16 = 2 байта
        wav.setframerate(rec.samplerate)

        frames_needed = None
        if rec.blocksize:  # blocksize == 0 → PortAudio выбирает размер сам
            frames_needed = int(seconds * rec.samplerate / rec.blocksize)

        for i, frame in enumerate(rec.frames()):
            wav.writeframes(frame.tobytes())
            if frames_needed is not None and i >= frames_needed:
                break


def get_loopback_capable_devices():
    """Возвращает список индексов устройств, поддерживающих loopback-захват."""
    try:
        wasapi_index = next(
            i
            for i, h in enumerate(sd.query_hostapis())
            if h["name"] == "Windows WASAPI"
        )
    except StopIteration:
        return []

    loopback_devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["hostapi"] == wasapi_index:
            if dev["max_input_channels"] > 0 and "loopback" in dev["name"].lower():
                loopback_devices.append(idx)
            elif dev["max_output_channels"] > 0:
                loopback_devices.append(idx)
    return loopback_devices


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Запись системного звука / выбранного input"
    )
    parser.add_argument("--seconds", type=int, default=5, help="длительность записи")
    parser.add_argument(
        "--outfile", type=str, default="out.wav", help="файл WAV вывода"
    )
    parser.add_argument(
        "--list", action="store_true", help="показать все устройства и выйти"
    )
    parser.add_argument("--device", type=int, help="индекс устройства (см. --list)")
    parser.add_argument(
        "--samplerate",
        type=int,
        help="частота дискретизации (по умолч. как у устройства)",
    )
    parser.add_argument(
        "--channels", type=int, help="каналов (1=mono,2=stereo). По умолч. авт."
    )
    args = parser.parse_args()

    if args.list:
        list_devices_verbose()
        return

    if DEVICE_INDEX:
        device_idx = DEVICE_INDEX
    else:
        device_idx = (
            args.device if args.device is not None else find_default_loopback_device()
        )

    # Override defaults if user provided.
    if args.samplerate:
        global DEFAULT_SAMPLE_RATE
        DEFAULT_SAMPLE_RATE = args.samplerate
    if args.channels:
        global DEFAULT_CHANNELS
        DEFAULT_CHANNELS = args.channels

    _record_cli(args.seconds, args.outfile, device_idx)


if __name__ == "__main__":  # pragma: no cover
    list_devices_verbose()
    main()
