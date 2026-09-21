"""Сборка полного скана: видео на входе, результат на выходе.

Модуль связывает остальные части ``core`` в один конвейер и держит всю
обработку ошибок на границе: что бы ни случилось внутри, наружу выходит
либо результат, либо исключение с текстом, понятным пользователю.

Порядок шагов важен. Качество оценивается ПОСЛЕ пульса, потому что одним
из его входов служит отношение сигнал/шум спектра, а оно появляется только
на этапе оценки частоты.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy.typing as npt

import config
from core import face, fatigue as fatigue_mod, quality as quality_mod, rppg, video_io

log = logging.getLogger(__name__)

#: Колбэк прогресса: (доля 0–1, подпись этапа).
ProgressFn = Callable[[float, str], None]


class ScanError(RuntimeError):
    """Скан не удался. Текст исключения показывается пользователю как есть."""


@dataclass
class ScanResult:
    """Полный результат одного замера."""

    heart_rate: rppg.HeartRateResult
    fatigue: fatigue_mod.FatigueResult
    quality: quality_mod.QualityReport
    preview: npt.NDArray | None
    duration_sec: float
    processing_sec: float
    created_at: datetime

    @property
    def bpm_display(self) -> str:
        """Пульс для показа. Прочерк, если качеству доверять нельзя."""
        if not self.quality.level.is_trustworthy:
            return "—"
        return f"{self.heart_rate.bpm:.0f}"


def run_scan(
    video_path: Path,
    progress: ProgressFn | None = None,
) -> ScanResult:
    """Прогоняет видео через весь конвейер.

    Args:
        video_path: путь к видеофайлу.
        progress: необязательный колбэк для прогресс-бара.

    Returns:
        ScanResult с пульсом, усталостью, качеством и кадром-превью.

    Raises:
        ScanError: с текстом, готовым к показу пользователю.
    """
    started = time.perf_counter()

    def report(fraction: float, label: str) -> None:
        if progress is not None:
            progress(fraction, label)

    # ── 1. Чтение видео и поиск лица ──
    report(0.05, "Читаю видео…")
    try:
        frames = video_io.iter_frames(video_path)
    except video_io.VideoReadError as exc:
        raise ScanError(str(exc)) from exc

    report(0.15, "Ищу лицо…")
    try:
        # Прогресс внутри обработки кадров: общее число кадров заранее
        # неизвестно (метаданные врут), поэтому ползём от 0.15 к 0.60
        # по приблизительной оценке в 600 кадров.
        def frame_progress(done: int, _total: object) -> None:
            report(min(0.15 + 0.45 * done / 600.0, 0.60), "Анализирую кровоток…")

        features = face.extract_features(frames, progress_callback=frame_progress)
    except face.NoFaceError as exc:
        raise ScanError(str(exc)) from exc
    except video_io.VideoReadError as exc:
        raise ScanError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — граница модуля, наружу не пускаем
        log.exception("Сбой при извлечении признаков")
        raise ScanError(
            "Не удалось обработать видео. Попробуйте записать заново."
        ) from exc

    # ── 2. Длительность ──
    try:
        video_io.validate_duration(features.timestamps.tolist())
    except video_io.VideoReadError as exc:
        raise ScanError(str(exc)) from exc

    duration = float(features.timestamps[-1] - features.timestamps[0])

    # ── 3. Пульс ──
    report(0.70, "Считаю пульс…")
    try:
        heart_rate = rppg.estimate_heart_rate(features.timestamps, features.rgb)
    except rppg.SignalTooShortError as exc:
        raise ScanError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Сбой оценки пульса")
        raise ScanError(
            "Не удалось выделить пульсовой сигнал. "
            "Нужен ровный свет спереди и неподвижная голова."
        ) from exc

    # ── 4. Усталость ──
    report(0.85, "Оцениваю усталость…")
    try:
        fatigue = fatigue_mod.analyze(features.landmarks, features.timestamps)
    except Exception as exc:  # noqa: BLE001
        # Усталость — не критичный модуль. Если он упал, пульс всё равно
        # показываем: потерять часть результата лучше, чем весь.
        log.exception("Сбой оценки усталости, продолжаю без неё")
        fatigue = fatigue_mod.FatigueResult(
            index=0.0,
            level=fatigue_mod.FatigueLevel.LOW,
            blink_rate_per_min=0.0,
            mean_blink_duration_ms=0.0,
            perclos=0.0,
            yawn_count=0,
            blink_count=0,
            duration_sec=duration,
            reliable=False,
        )

    # ── 5. Качество ──
    report(0.95, "Проверяю качество…")
    quality = quality_mod.assess(features, heart_rate.snr_db)

    elapsed = time.perf_counter() - started
    report(1.0, "Готово")

    log.info(
        "Скан завершён за %.1f с: пульс %.0f, усталость %.0f, качество %s",
        elapsed, heart_rate.bpm, fatigue.index, quality.level.value,
    )

    return ScanResult(
        heart_rate=heart_rate,
        fatigue=fatigue,
        quality=quality,
        preview=features.preview,
        duration_sec=duration,
        processing_sec=elapsed,
        created_at=datetime.now(),
    )


def find_demo_video() -> Path | None:
    """Ищет демо-видео в папке demo/.

    Запасной путь для сцены: если камера или загрузка подведут,
    показываем заранее записанный ролик.
    """
    if not config.DEMO_DIR.exists():
        return None

    for pattern in ("*.mp4", "*.mov", "*.MOV", "*.webm", "*.m4v"):
        found = sorted(config.DEMO_DIR.glob(pattern))
        if found:
            return found[0]

    return None
