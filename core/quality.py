"""Оценка качества сигнала.

Самое опасное поведение для такого приложения — уверенно показать число,
полученное из шума. Пульс, посчитанный по тёмному дёрганому видео, выглядит
ровно так же убедительно, как настоящий, и отличить их на глаз невозможно.

Поэтому качество считается отдельно и имеет право вето: при красном статусе
конкретное число пульса пользователю не показывается вообще.

Смотрим на четыре вещи:
    * долю кадров, где нашлось лицо;
    * движение головы, нормированное на размер лица;
    * отношение сигнал/шум спектра пульса;
    * яркость ROI — темнота и пересвет одинаково вредны.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

import config
from core.face import ScanFeatures

log = logging.getLogger(__name__)


class QualityLevel(str, Enum):
    """Итоговый статус замера."""

    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

    @property
    def emoji(self) -> str:
        return {"good": "🟢", "fair": "🟡", "poor": "🔴"}[self.value]

    @property
    def label(self) -> str:
        return {"good": "Хорошее", "fair": "Среднее", "poor": "Плохое"}[self.value]

    @property
    def is_trustworthy(self) -> bool:
        """Можно ли показывать пользователю конкретное число пульса."""
        return self is not QualityLevel.POOR


@dataclass(frozen=True)
class QualityReport:
    """Разбор качества записи."""

    level: QualityLevel
    face_ratio: float
    motion: float
    """Среднее смещение центра лица за кадр в долях размера лица."""

    snr_db: float
    brightness: float
    hints: tuple[str, ...]
    """Подсказки пользователю — что именно исправить при пересъёмке."""

    @property
    def summary(self) -> str:
        return f"{self.level.emoji} {self.level.label}"


def _motion_score(features: ScanFeatures) -> float:
    """Насколько сильно двигалась голова.

    Смещение нормируем на размер лица: иначе тот, кто сидит ближе к камере,
    всегда выглядел бы дёрганым просто из-за масштаба.
    """
    centers = features.face_centers
    sizes = features.face_sizes

    if centers.shape[0] < 2:
        return 0.0

    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    scale = np.maximum(sizes[1:], 1e-6)
    normalized = steps / scale

    # Face Mesh запускается не на каждом кадре, поэтому часть соседних пар
    # содержит одни и те же ландмарки и даёт смещение ровно 0. Если их учесть,
    # медиана схлопнется в ноль и любая тряска будет выглядеть как штатив.
    moved = normalized[normalized > 0.0]
    if moved.size == 0:
        return 0.0

    return float(np.median(moved))


def _brightness(features: ScanFeatures) -> float:
    """Средняя яркость ROI в шкале 0–255."""
    if features.rgb.size == 0:
        return 0.0
    return float(features.rgb.mean())


def assess(features: ScanFeatures, snr_db: float) -> QualityReport:
    """Собирает итоговую оценку качества.

    Правило простое: итог равен худшему из отдельных показателей.
    Хорошая освещённость не компенсирует тряску, а неподвижность —
    потерянное лицо.

    Args:
        features: признаки, собранные модулем ``face``.
        snr_db: отношение сигнал/шум из оценки пульса.

    Returns:
        QualityReport со статусом и подсказками.
    """
    face_ratio = features.face_ratio
    motion = _motion_score(features)
    brightness = _brightness(features)

    levels: list[QualityLevel] = []
    hints: list[str] = []

    # ── Лицо в кадре ──
    if face_ratio >= config.QUALITY_MIN_FACE_RATIO_GOOD:
        levels.append(QualityLevel.GOOD)
    elif face_ratio >= config.QUALITY_MIN_FACE_RATIO_OK:
        levels.append(QualityLevel.FAIR)
        hints.append("Лицо иногда выходило из кадра — держите телефон ровнее.")
    else:
        levels.append(QualityLevel.POOR)
        hints.append("Лицо потерялось на большей части записи. Снимите ещё раз.")

    # ── Движение ──
    if motion <= config.QUALITY_MAX_MOTION_GOOD:
        levels.append(QualityLevel.GOOD)
    elif motion <= config.QUALITY_MAX_MOTION_OK:
        levels.append(QualityLevel.FAIR)
        hints.append("Небольшое движение головы. Постарайтесь сидеть неподвижно.")
    else:
        levels.append(QualityLevel.POOR)
        hints.append("Вы двигались во время записи — повторите замер неподвижно.")

    # ── Отношение сигнал/шум ──
    if snr_db >= config.QUALITY_MIN_SNR_GOOD:
        levels.append(QualityLevel.GOOD)
    elif snr_db >= config.QUALITY_MIN_SNR_OK:
        levels.append(QualityLevel.FAIR)
        hints.append("Сигнал слабоват — результат приблизительный.")
    else:
        levels.append(QualityLevel.POOR)
        hints.append("Пульсовой сигнал не выделяется из шума. Нужна пересъёмка.")

    # ── Освещение ──
    dark, bright = config.QUALITY_BRIGHTNESS_RANGE
    if brightness < dark:
        levels.append(QualityLevel.POOR if brightness < dark * 0.6 else QualityLevel.FAIR)
        hints.append("Слишком темно. Добавьте света спереди — например, от окна.")
    elif brightness > bright:
        levels.append(QualityLevel.FAIR)
        hints.append("Пересвет. Уберите яркий источник света прямо перед лицом.")
    else:
        levels.append(QualityLevel.GOOD)

    # Итог — по худшему показателю.
    if QualityLevel.POOR in levels:
        level = QualityLevel.POOR
    elif QualityLevel.FAIR in levels:
        level = QualityLevel.FAIR
    else:
        level = QualityLevel.GOOD
        hints.append("Отличная запись — результату можно доверять.")

    log.info(
        "Качество %s: лицо %.0f%%, движение %.4f, SNR %.1f дБ, яркость %.0f",
        level.value, face_ratio * 100, motion, snr_db, brightness,
    )

    return QualityReport(
        level=level,
        face_ratio=face_ratio,
        motion=motion,
        snr_db=snr_db,
        brightness=brightness,
        hints=tuple(hints),
    )
