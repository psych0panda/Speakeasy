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
from typing import Generator, Optional, cast

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

# Целевой уровень громкости (RMS) в дБFS. 0 dBFS = full-scale (32768).
# Значение −20 dBFS считается комфортным для речи.
DEFAULT_TARGET_RMS_DBFS = -20.0


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def list_devices_verbose() -> None:
    """Выводит таблицу со всеми аудио-устройствами."""
    for idx, dev_raw in enumerate(sd.query_devices()):
        dev: dict[str, object] = dict(dev_raw)  # type: ignore[arg-type]
        ha = sd.query_hostapis(dev["hostapi"])  # type: ignore[index]
        ha_name = ha["name"] if isinstance(ha, dict) else ha.name  # type: ignore[index]
        print(
            f"[{idx:2}] {dev['name']} | in {dev['max_input_channels']} "
            f"out {dev['max_output_channels']} | {ha_name}"
        )


def find_default_loopback_device() -> Optional[int]:
    """Ищем устройство для системного loopback-захвата.

    Таким устройством является "CABLE Output (VB-Audio Virtual Cable)"
    поскольку оно регистрируется как output-устройство, но при этом
    поддерживает loopback-захват.
    """

    for idx, dev_raw in enumerate(sd.query_devices()):
        dev: dict[str, object] = dict(dev_raw)  # type: ignore[arg-type]
        ha = sd.query_hostapis(dev["hostapi"])  # type: ignore[index]
        ha_name = ha["name"] if isinstance(ha, dict) else ha.name  # type: ignore[index]
        if dev["name"] == "CABLE Output (VB-Audio Virtual Cable)":
            if dev["max_input_channels"] == 2 and dev["max_output_channels"] == 0:
                if ha_name == "Windows WASAPI":
                    print(f"Выбранное устройство: {dev['name']}")
                    return int(dev["index"])  # type: ignore[index]

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
        target_rms_dbfs: float | None = None,
        auto_restart: bool = True,
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
        self._auto_restart = auto_restart

        # Нормализация
        self._normalize = target_rms_dbfs is not None
        self._target_rms_linear: float | None = None
        if self._normalize:
            # dBFS → линейный коэффициент (0..1)
            self._target_rms_linear = 10.0 ** (target_rms_dbfs / 20.0)

        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=20)

        # Реальный PortAudio stream создаём лениво в .start(),
        # чтобы не занимать устройство зря, если объект только конфигурируется.
        self._stream: sd.InputStream | None = None

        # Импорт здесь, чтобы не тянуть во все случаи
        import time as _time  # noqa: WPS433

        self._time = _time

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
            # Авто-рестарт, если включён и поток неожиданно остановлен
            if self._auto_restart and (self._stream is None or not self._stream.active):
                try:
                    self.restart()
                except Exception as exc:  # noqa: BLE001
                    print(f"[LoopbackRecorder] restart failed: {exc}", file=sys.stderr)
                    self._time.sleep(1)
                    continue

            data = self._q.get()
            yield data

    # ------------------------------------------------------------------
    # Internal callback
    # ------------------------------------------------------------------
    def _callback(self, indata, frames, time, status):  # noqa: N802
        if status:
            # Выводим, но не спамим слишком часто
            print(f"[LoopbackRecorder] status: {status}", file=sys.stderr)

        pcm = indata.copy()

        # RMS-нормализация по желанию пользователя
        if self._normalize and self._target_rms_linear:
            # Переводим в float32 в диапазон [-1..1]
            pcm_f = pcm.astype("float32") / 32768.0
            rms = np.sqrt(np.mean(np.square(pcm_f)))
            if rms > 0:
                gain = self._target_rms_linear / rms
                # ограничиваем усиление, чтобы не взорваться громкостью
                gain = min(gain, 20.0)  # +26 dB max
                pcm_f *= gain
                pcm_f = np.clip(pcm_f, -1.0, 1.0)
                pcm = (pcm_f * 32768.0).astype(DTYPE)

        self._q.put(pcm)

    def restart(self) -> None:
        """Перезапускает поток (используется при ошибках устройства)."""
        try:
            self.stop()
        except Exception:  # noqa: BLE001
            pass
        self.start()


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
    assert outfile is not None, "outfile path должен быть строкой"
    wav_file = cast(wave.Wave_write, wave.open(outfile, "wb"))
    with LoopbackRecorder(device=device_idx) as rec, wav_file as wav:
        # задаём параметры выходного WAV-файла
        wav.setnchannels(int(rec.channels or 1))  # type: ignore[attr-defined]
        wav.setsampwidth(2)  # type: ignore[attr-defined]
        wav.setframerate(rec.samplerate)  # type: ignore[attr-defined]

        frames_needed = None
        if rec.blocksize and rec.blocksize > 0:  # blocksize == 0 → авто
            frames_needed = int(seconds * rec.samplerate // rec.blocksize)

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
            if str(dict(h).get("name", "")) == "Windows WASAPI"  # type: ignore[index]
        )
    except StopIteration:
        return []

    loopback_devices: list[int] = []
    for idx, dev_raw in enumerate(sd.query_devices()):
        dev: dict[str, object] = dict(dev_raw)  # type: ignore[arg-type]
        ha_raw = sd.query_hostapis(dev["hostapi"])  # type: ignore[index]
        ha_dict: dict[str, object] = (
            dict(ha_raw) if not isinstance(ha_raw, dict) else ha_raw
        )  # type: ignore[arg-type]
        ha_name = str(ha_dict.get("name", ha_raw))
        if int(dev["hostapi"]) == wasapi_index:  # type: ignore[index]
            if (
                int(dev["max_input_channels"]) > 0
                and "loopback" in str(dev["name"]).lower()
            ):
                loopback_devices.append(idx)
            elif int(dev["max_output_channels"]) > 0:
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
    parser.add_argument(
        "--norm",
        action="store_true",
        help="включить RMS-нормализацию до −20 dBFS",
    )
    parser.add_argument(
        "--norm-level",
        type=float,
        default=-20.0,
        help="целевой уровень RMS dBFS (используется с --norm)",
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

    global DEFAULT_TARGET_RMS_DBFS
    target_dbfs = args.norm_level if args.norm else None

    rec = LoopbackRecorder(
        device=device_idx,
        samplerate=DEFAULT_SAMPLE_RATE,
        channels=DEFAULT_CHANNELS,
        target_rms_dbfs=target_dbfs,
    )

    _record_cli(args.seconds, args.outfile, device_idx)


if __name__ == "__main__":  # pragma: no cover
    list_devices_verbose()
    main()
