"""Захват звука с микрофона и непрерывное распознавание.

Устроено как долгоживущий объект с фоновым потоком, а не как функция.
Причина в Streamlit: он перезапускает скрипт на каждое действие
пользователя, и всё, что живёт в переменных, умирает. Поток и накопленная
расшифровка должны пережить десятки перерисовок, поэтому состояние
держится здесь, а интерфейс только читает его.

Поток данных:

    микрофон → очередь → фоновый поток → Vosk → накопленный текст
                                                        ↓
                                              интерфейс читает и рисует

Очередь между микрофоном и распознаванием нужна обязательно: обратный
вызов звуковой карты обязан возвращаться мгновенно. Если распознавать
прямо в нём, звук начнёт заикаться и куски будут теряться.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import config
from core.asr import AsrError, DualTranscriber

log = logging.getLogger(__name__)


class AudioError(RuntimeError):
    """Не удалось получить звук. Текст готов к показу пользователю."""


@dataclass
class ListenerState:
    """Снимок состояния для интерфейса."""

    running: bool = False
    transcript: str = ""
    partial: str = ""
    """Текущая незаконченная фраза — она ещё может измениться."""

    seconds: float = 0.0
    error: str = ""
    chunks: int = 0
    """Сколько кусков звука обработано — видно, что поток жив."""

    levels: list[float] = field(default_factory=list)
    """Последние уровни громкости для индикатора."""


class MicrophoneListener:
    """Слушает микрофон и непрерывно расшифровывает речь.

    Потокобезопасен: интерфейс дёргает ``snapshot`` из главного потока,
    пока фоновый пишет. Всё общее состояние под одним замком.
    """

    def __init__(self, device: int | None = None) -> None:
        self._device = device
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=config.AUDIO_QUEUE_MAX)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stream = None

        self._transcriber: DualTranscriber | None = None
        self._partial = ""
        self._started_at = 0.0
        self._chunks = 0
        self._error = ""
        self._levels: list[float] = []

    # ── Управление ──

    def start(self) -> None:
        """Открывает микрофон и запускает распознавание.

        Raises:
            AudioError: нет микрофона или моделей.
        """
        if self.running:
            return

        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover
            raise AudioError("Не установлен sounddevice.") from exc

        try:
            self._transcriber = DualTranscriber()
        except AsrError as exc:
            raise AudioError(str(exc)) from exc

        self._stop.clear()
        self._error = ""
        self._chunks = 0
        self._partial = ""
        self._levels = []
        self._started_at = time.monotonic()

        def on_audio(indata, frames, time_info, status) -> None:
            """Обратный вызов звуковой карты. Обязан быть быстрым."""
            if status:
                log.debug("Статус звукового потока: %s", status)
            try:
                self._queue.put_nowait(bytes(indata))
            except queue.Full:
                # Очередь переполнена — распознавание отстаёт. Роняем кусок:
                # потерять четверть секунды лучше, чем копить задержку,
                # которая к концу звонка вырастет до минут.
                log.warning("Очередь звука переполнена, кусок отброшен")

        try:
            self._stream = sd.RawInputStream(
                samplerate=config.ASR_SAMPLE_RATE,
                blocksize=config.ASR_CHUNK_FRAMES,
                device=self._device,
                dtype="int16",
                channels=1,
                callback=on_audio,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 — сообщения звуковых систем непредсказуемы
            self._transcriber = None
            raise AudioError(
                "Не удалось открыть микрофон. Проверьте разрешение "
                "в настройках системы и что его не занял другой звонок.\n\n"
                f"Подробности: {exc}"
            ) from exc

        self._thread = threading.Thread(target=self._work, daemon=True, name="asr")
        self._thread.start()
        log.info("Слушаю микрофон")

    def stop(self) -> None:
        """Останавливает захват, сохраняя накопленную расшифровку."""
        self._stop.set()

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("Сбой при закрытии потока: %s", exc)
            self._stream = None

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            if self._transcriber is not None:
                self._transcriber.flush()
            self._partial = ""

        log.info("Микрофон остановлен")

    def reset(self) -> None:
        """Начинает новый разговор с чистого листа."""
        self.stop()
        with self._lock:
            if self._transcriber is not None:
                self._transcriber.reset()
            self._partial = ""
            self._chunks = 0
            self._levels = []

    # ── Фоновая работа ──

    def _work(self) -> None:
        """Тянет звук из очереди и скармливает распознавателю."""
        while not self._stop.is_set():
            try:
                pcm = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                with self._lock:
                    if self._transcriber is None:
                        continue
                    ru, _kk = self._transcriber.feed(pcm)
                    self._partial = ru.text if not ru.is_final else ""
                    self._chunks += 1
                    self._levels.append(_rms(pcm))
                    if len(self._levels) > config.AUDIO_LEVEL_HISTORY:
                        self._levels = self._levels[-config.AUDIO_LEVEL_HISTORY :]
            except Exception as exc:  # noqa: BLE001 — поток не должен умирать молча
                log.exception("Сбой распознавания")
                with self._lock:
                    self._error = f"Сбой распознавания: {exc}"
                break

    # ── Чтение состояния ──

    @property
    def running(self) -> bool:
        return self._stream is not None and not self._stop.is_set()

    def snapshot(self) -> ListenerState:
        """Согласованный снимок состояния для интерфейса."""
        with self._lock:
            transcript = self._transcriber.transcript if self._transcriber else ""
            return ListenerState(
                running=self.running,
                transcript=transcript,
                partial=self._partial,
                seconds=time.monotonic() - self._started_at if self._started_at else 0.0,
                error=self._error,
                chunks=self._chunks,
                levels=list(self._levels),
            )


def _rms(pcm: bytes) -> float:
    """Громкость куска в долях от максимума — для индикатора уровня."""
    import array
    import math

    samples = array.array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0.0
    total = sum(s * s for s in samples)
    return min(math.sqrt(total / len(samples)) / 32768.0 * 4.0, 1.0)


def list_input_devices() -> list[tuple[int, str]]:
    """Перечисляет микрофоны — чтобы на сцене выбрать нужный."""
    try:
        import sounddevice as sd
    except ImportError:
        return []
    return [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]
