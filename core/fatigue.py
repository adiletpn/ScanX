"""Признаки усталости по мимике.

Считаем четыре величины и сводим их в один индекс:

* **EAR** (Eye Aspect Ratio) — отношение высоты глаза к ширине.
  Падает почти до нуля в момент моргания.
* **Частота морганий** — при утомлении растёт.
* **PERCLOS** — доля времени с закрытыми глазами. В исследованиях сонливости
  это самый устойчивый из простых показателей.
* **Зевки** по MAR (Mouth Aspect Ratio).

Важно: порог моргания берётся ОТНОСИТЕЛЬНО медианного EAR конкретного
человека. Разрез глаз у всех разный, и фиксированный порог у одних
насчитает моргание на каждом кадре, а у других не заметит ни одного.

Ограничение, которое честно проговариваем в интерфейсе: 20 секунд — короткое
окно. Индекс здесь оценочный и служит поводом задуматься, а не выводом.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

import config

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]


class FatigueLevel(str, Enum):
    """Уровень усталости."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

    @property
    def label(self) -> str:
        return {"low": "Низкий", "moderate": "Умеренный", "high": "Высокий"}[self.value]

    @property
    def color(self) -> str:
        return {
            "low": config.COLOR_GREEN,
            "moderate": config.COLOR_AMBER,
            "high": config.COLOR_RED,
        }[self.value]


@dataclass(frozen=True)
class FatigueResult:
    """Результат оценки усталости."""

    index: float
    """Итоговый индекс 0–100. Оценочный."""

    level: FatigueLevel
    blink_rate_per_min: float
    mean_blink_duration_ms: float
    perclos: float
    """Доля времени с закрытыми глазами, 0–1."""

    yawn_count: int
    blink_count: int
    duration_sec: float
    reliable: bool
    """False, если запись слишком короткая для осмысленных выводов."""


def _eye_aspect_ratio(points: FloatArray, indices: list[int]) -> float:
    """Считает EAR по шести точкам глаза.

    Классическая формула (Soukupová & Čech, 2016):

        EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2·‖p1−p4‖)

    В числителе — две вертикальные хорды, в знаменателе — горизонтальная
    ширина глаза. Деление на ширину делает величину безразмерной, поэтому
    она не зависит от расстояния до камеры.
    """
    p = points[indices]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / (2.0 * horizontal))


def _mouth_aspect_ratio(points: FloatArray) -> float:
    """MAR: раскрытие рта по вертикали, отнесённое к ширине."""
    left, right = points[config.MOUTH_CORNERS[0]], points[config.MOUTH_CORNERS[1]]
    top, bottom = points[config.MOUTH_VERTICAL[0]], points[config.MOUTH_VERTICAL[1]]

    width = np.linalg.norm(left - right)
    if width < 1e-6:
        return 0.0
    return float(np.linalg.norm(top - bottom) / width)


def _find_episodes(flags: npt.NDArray[np.bool_], min_length: int) -> list[tuple[int, int]]:
    """Ищет непрерывные серии True длиной не меньше ``min_length``.

    Используется и для морганий, и для зевков: в обоих случаях нас интересует
    не отдельный кадр, а эпизод. Порог по длине отсекает одиночные выбросы
    детектора, которые иначе засчитались бы как моргание.

    Returns:
        Список пар (начало, конец), конец не включается.
    """
    if flags.size == 0:
        return []

    episodes: list[tuple[int, int]] = []
    start: int | None = None

    for i, active in enumerate(flags):
        if active and start is None:
            start = i
        elif not active and start is not None:
            if i - start >= min_length:
                episodes.append((start, i))
            start = None

    if start is not None and flags.size - start >= min_length:
        episodes.append((start, flags.size))

    return episodes


def analyze(
    landmarks: FloatArray,
    timestamps: FloatArray,
) -> FatigueResult:
    """Считает признаки усталости по ландмаркам всех кадров.

    Формула индекса (0–100) — взвешенная сумма четырёх вкладов.
    Веса расставлены по тому, насколько показатель устойчив на коротком
    двадцатисекундном окне:

        PERCLOS              40  — самый надёжный маркер сонливости
        Частота морганий     25  — растёт при утомлении
        Длительность морганий 20 — «тяжёлые» веки закрываются дольше
        Зевки                15  — яркий, но редкий на 20 секундах признак

    Каждый вклад нормируется в диапазон 0–1 и умножается на свой вес.

    Args:
        landmarks: массив (N, K, 2) ландмарок по кадрам.
        timestamps: метки времени кадров, секунды.

    Returns:
        FatigueResult с индексом, уровнем и отдельными показателями.
    """
    n_frames = landmarks.shape[0]
    duration = float(timestamps[-1] - timestamps[0]) if n_frames > 1 else 0.0

    if n_frames < 10 or duration < 3.0:
        return FatigueResult(
            index=0.0,
            level=FatigueLevel.LOW,
            blink_rate_per_min=0.0,
            mean_blink_duration_ms=0.0,
            perclos=0.0,
            yawn_count=0,
            blink_count=0,
            duration_sec=duration,
            reliable=False,
        )

    ear = np.array(
        [
            (
                _eye_aspect_ratio(frame, config.EYE_LEFT)
                + _eye_aspect_ratio(frame, config.EYE_RIGHT)
            )
            / 2.0
            for frame in landmarks
        ],
        dtype=np.float64,
    )
    mar = np.array([_mouth_aspect_ratio(frame) for frame in landmarks], dtype=np.float64)

    # Порог считаем от медианы этого человека: медиана устойчива к
    # выбросам, а моргания как раз и есть выбросы вниз.
    baseline_ear = float(np.median(ear))
    threshold = baseline_ear * config.BLINK_EAR_RATIO

    closed = ear < threshold

    # PERCLOS — просто доля закрытых кадров.
    perclos = float(closed.mean())

    blinks = _find_episodes(closed, config.BLINK_MIN_FRAMES)
    blink_count = len(blinks)
    blink_rate = blink_count / duration * 60.0 if duration > 0 else 0.0

    if blinks:
        durations = [
            float(timestamps[min(end, n_frames - 1)] - timestamps[start])
            for start, end in blinks
        ]
        mean_blink_ms = float(np.mean(durations) * 1000.0)
    else:
        mean_blink_ms = 0.0

    yawn_frames = mar > config.YAWN_MAR_THRESHOLD
    # Зевок — не просто открытый рот, а открытый надолго. Иначе речь
    # и любая гримаса засчитались бы как зевок.
    fps_estimate = n_frames / duration if duration > 0 else 30.0
    min_yawn_frames = max(int(config.YAWN_MIN_DURATION_SEC * fps_estimate), 2)
    yawn_count = len(_find_episodes(yawn_frames, min_yawn_frames))

    # ── Сборка индекса ──

    # PERCLOS: 5% — норма бодрствующего человека, 35% — выраженная сонливость.
    perclos_score = np.clip(
        (perclos - 0.05) / (0.35 - 0.05), 0.0, 1.0
    )

    # Частота морганий: норма 12–20 в минуту. Вклад даёт превышение
    # верхней границы; у сонного человека моргания учащаются.
    _, rate_high = config.BLINK_RATE_NORMAL_RANGE
    rate_score = np.clip((blink_rate - rate_high) / 20.0, 0.0, 1.0)

    # Длительность моргания: обычное — около 150 мс, вялое — 400 мс и дольше.
    duration_score = np.clip((mean_blink_ms - 150.0) / (400.0 - 150.0), 0.0, 1.0)

    # Зевки: один зевок за 20 секунд — уже много.
    yawn_score = np.clip(yawn_count / 2.0, 0.0, 1.0)

    index = float(
        perclos_score * 40.0
        + rate_score * 25.0
        + duration_score * 20.0
        + yawn_score * 15.0
    )
    index = float(np.clip(index, 0.0, 100.0))

    if index <= config.FATIGUE_LOW_MAX:
        level = FatigueLevel.LOW
    elif index <= config.FATIGUE_MODERATE_MAX:
        level = FatigueLevel.MODERATE
    else:
        level = FatigueLevel.HIGH

    log.info(
        "Усталость %.0f (%s): PERCLOS %.2f, морганий %.1f/мин, зевков %d",
        index, level.value, perclos, blink_rate, yawn_count,
    )

    return FatigueResult(
        index=index,
        level=level,
        blink_rate_per_min=blink_rate,
        mean_blink_duration_ms=mean_blink_ms,
        perclos=perclos,
        yawn_count=yawn_count,
        blink_count=blink_count,
        duration_sec=duration,
        # 20 секунд — нижняя граница осмысленности. Меньше 15 —
        # моргания просто не успевают набрать статистику.
        reliable=duration >= 15.0,
    )
