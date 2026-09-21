"""Офлайн-распознавание речи через Vosk.

Почему Vosk, а не Whisper:

* **Потоковость.** Vosk отдаёт результат по мере речи, ещё до конца фразы.
  Whisper требует готовый отрезок. Для «поймать мошенника посреди разговора»
  это решающее свойство: предупредить нужно ДО того, как человек назовёт код.
* **Казахский обучен отдельно.** У Whisper казахский есть в списке языков,
  но он низкоресурсный и качество заметно хуже.
* **Размер.** 44 и 57 МБ против гигабайтов.
* **Скорость.** Реальное время на обычном процессоре, без видеокарты.

Качество маленьких моделей ограничено, и на казахском особенно. Но нам
не нужна точная расшифровка — движок ``scam_engine`` работает по корням
слов, а опорные слова схемы («банк», «код», «аудар») мошеннику придётся
произнести в любом случае.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import config

log = logging.getLogger(__name__)


class AsrError(RuntimeError):
    """Не удалось подготовить распознавание. Текст готов к показу."""


@dataclass
class Segment:
    """Кусок расшифровки."""

    text: str
    is_final: bool
    """False — промежуточный результат, он ещё может измениться."""


class Transcriber:
    """Потоковое распознавание речи одной языковой моделью.

    Работает по кускам: скармливаем аудио, получаем промежуточный текст,
    в конце фразы — окончательный. Промежуточный нужен, чтобы экран жил
    и человек видел, что система слушает.
    """

    def __init__(self, model_dir: Path, sample_rate: int = config.ASR_SAMPLE_RATE) -> None:
        try:
            import vosk
        except ImportError as exc:  # pragma: no cover — окружение, не логика
            raise AsrError(
                "Не установлен vosk. Выполните: pip install vosk"
            ) from exc

        if not model_dir.exists():
            raise AsrError(
                f"Нет речевой модели: {model_dir.name}\n"
                f"Скачайте её с alphacephei.com/vosk/models и распакуйте в models/"
            )

        # Vosk по умолчанию сыплет отладкой в stderr на каждый кусок аудио.
        vosk.SetLogLevel(-1)

        self._sample_rate = sample_rate
        self._model = vosk.Model(str(model_dir))
        self._rec = vosk.KaldiRecognizer(self._model, sample_rate)
        self._finals: list[str] = []
        log.info("Загружена речевая модель %s", model_dir.name)

    def feed(self, pcm: bytes) -> Segment:
        """Принимает кусок звука (16 бит, моно) и возвращает расшифровку.

        Args:
            pcm: сырые отсчёты int16 в байтах.

        Returns:
            Segment с текстом. ``is_final`` — фраза завершена.
        """
        if self._rec.AcceptWaveform(pcm):
            text = json.loads(self._rec.Result()).get("text", "").strip()
            if text:
                self._finals.append(text)
            return Segment(text=text, is_final=True)

        partial = json.loads(self._rec.PartialResult()).get("partial", "").strip()
        return Segment(text=partial, is_final=False)

    def flush(self) -> str:
        """Забирает остаток после конца записи."""
        text = json.loads(self._rec.FinalResult()).get("text", "").strip()
        if text:
            self._finals.append(text)
        return text

    @property
    def transcript(self) -> str:
        """Весь распознанный текст с начала разговора."""
        return " ".join(self._finals)

    def reset(self) -> None:
        """Начинает новый разговор."""
        import vosk

        self._rec = vosk.KaldiRecognizer(self._model, self._sample_rate)
        self._finals = []


class DualTranscriber:
    """Распознавание сразу на двух языках.

    Мошенники в Казахстане мешают русский и казахский прямо внутри
    разговора — «здравствуйте, банк қызметкері, срочно кодты айтыңыз».
    Выбирать язык заранее значит терять половину фраз.

    Поэтому звук идёт в обе модели, а движок схем получает **объединённый**
    текст. Лишние слова ему не мешают: он ищет корни, а не разбирает
    предложения. Зато ни одна опорная фраза не теряется.

    Цена — вдвое больше вычислений. На обычном ноутбуке это всё равно
    быстрее реального времени.
    """

    def __init__(self) -> None:
        self._ru = Transcriber(config.ASR_MODEL_RU)
        self._kk: Transcriber | None = None
        try:
            self._kk = Transcriber(config.ASR_MODEL_KK)
        except AsrError as exc:
            # Без казахского работать можно, без русского — нет.
            log.warning("Казахская модель недоступна: %s", exc)

    def feed(self, pcm: bytes) -> tuple[Segment, Segment | None]:
        """Скармливает кусок обеим моделям."""
        ru = self._ru.feed(pcm)
        kk = self._kk.feed(pcm) if self._kk is not None else None
        return ru, kk

    def flush(self) -> None:
        self._ru.flush()
        if self._kk is not None:
            self._kk.flush()

    @property
    def transcript(self) -> str:
        """Объединённый текст для движка схем."""
        parts = [self._ru.transcript]
        if self._kk is not None:
            parts.append(self._kk.transcript)
        return " ".join(p for p in parts if p)

    @property
    def transcript_ru(self) -> str:
        return self._ru.transcript

    @property
    def transcript_kk(self) -> str:
        return self._kk.transcript if self._kk is not None else ""

    @property
    def has_kazakh(self) -> bool:
        return self._kk is not None

    def reset(self) -> None:
        self._ru.reset()
        if self._kk is not None:
            self._kk.reset()


def transcribe_file(path: Path, dual: bool = True) -> str:
    """Расшифровывает готовый аудиофайл целиком.

    Нужен для двух вещей: проверки движка на записанных сценариях
    и запасного варианта на сцене, если живой звук подведёт.

    Args:
        path: путь к WAV (16 бит, моно; другое перекодируется через ffmpeg).
        dual: распознавать обоими языками.

    Raises:
        AsrError: файл не читается или нет моделей.
    """
    import subprocess
    import shutil
    import wave

    path = Path(path)
    if not path.exists():
        raise AsrError(f"Файл не найден: {path}")

    # Приводим к тому, что ждёт Vosk: 16 кГц, моно, 16 бит.
    prepared = path
    try:
        with wave.open(str(path), "rb") as wf:
            needs_convert = (
                wf.getnchannels() != 1
                or wf.getsampwidth() != 2
                or wf.getframerate() != config.ASR_SAMPLE_RATE
            )
    except Exception:
        needs_convert = True

    if needs_convert:
        if not shutil.which("ffmpeg"):
            raise AsrError(
                "Файл не в нужном формате, а ffmpeg не установлен. "
                "Нужен WAV 16 кГц моно."
            )
        prepared = path.with_suffix(".16k.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-ar", str(config.ASR_SAMPLE_RATE), "-ac", "1", "-f", "wav",
             str(prepared)],
            check=True, capture_output=True, timeout=120,
        )

    engine = DualTranscriber() if dual else Transcriber(config.ASR_MODEL_RU)

    with wave.open(str(prepared), "rb") as wf:
        while True:
            data = wf.readframes(config.ASR_CHUNK_FRAMES)
            if not data:
                break
            engine.feed(data)

    engine.flush()
    return engine.transcript
