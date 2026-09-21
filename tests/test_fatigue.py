"""Тесты модуля усталости на синтетических ландмарках.

Строим искусственное «лицо», у которого глаза закрываются по расписанию.
Так мы точно знаем, сколько морганий должно быть найдено, и можем проверить,
что относительный порог работает для разного разреза глаз.
"""

from __future__ import annotations

import numpy as np

import config
from core.fatigue import FatigueLevel, analyze


def _make_face(eye_openness: float, mouth_openness: float = 0.1) -> np.ndarray:
    """Собирает один кадр ландмарок с заданным раскрытием глаз и рта.

    Заполняем массив нулями и расставляем только те точки, которые
    действительно читает модуль: шесть на каждый глаз, четыре на рот
    и четыре опорные точки лица.

    Args:
        eye_openness: высота глаза в пикселях; ширина всегда 30.
        mouth_openness: раскрытие рта в долях ширины.
    """
    points = np.zeros((478, 2), dtype=np.float64)

    # Глаза: порядок точек [внешний угол, верх-1, верх-2, внутренний угол,
    # низ-2, низ-1] — тот же, что в config.
    for indices, x0 in ((config.EYE_LEFT, 100.0), (config.EYE_RIGHT, 200.0)):
        half = eye_openness / 2.0
        points[indices[0]] = (x0, 100.0)            # внешний угол
        points[indices[1]] = (x0 + 10.0, 100.0 - half)
        points[indices[2]] = (x0 + 20.0, 100.0 - half)
        points[indices[3]] = (x0 + 30.0, 100.0)     # внутренний угол
        points[indices[4]] = (x0 + 20.0, 100.0 + half)
        points[indices[5]] = (x0 + 10.0, 100.0 + half)

    # Рот: ширина 60 px, раскрытие задаётся в её долях.
    points[config.MOUTH_CORNERS[0]] = (120.0, 200.0)
    points[config.MOUTH_CORNERS[1]] = (180.0, 200.0)
    points[config.MOUTH_VERTICAL[0]] = (150.0, 200.0 - 60.0 * mouth_openness / 2.0)
    points[config.MOUTH_VERTICAL[1]] = (150.0, 200.0 + 60.0 * mouth_openness / 2.0)

    # Опорные точки лица.
    points[config.FACE_ANCHOR_TOP] = (150.0, 50.0)
    points[config.FACE_ANCHOR_BOTTOM] = (150.0, 250.0)
    points[config.FACE_ANCHOR_LEFT] = (80.0, 150.0)
    points[config.FACE_ANCHOR_RIGHT] = (220.0, 150.0)

    return points


def _make_sequence(
    n_frames: int = 600,
    fps: float = 30.0,
    open_height: float = 12.0,
    blink_every: int = 100,
    blink_frames: int = 4,
    yawn_at: int | None = None,
    yawn_frames: int = 45,
) -> tuple[np.ndarray, np.ndarray]:
    """Строит последовательность кадров с морганиями по расписанию."""
    frames = []
    for i in range(n_frames):
        blinking = (i % blink_every) < blink_frames
        height = open_height * 0.15 if blinking else open_height

        yawning = yawn_at is not None and yawn_at <= i < yawn_at + yawn_frames
        mouth = 0.9 if yawning else 0.1

        frames.append(_make_face(height, mouth))

    timestamps = np.arange(n_frames, dtype=np.float64) / fps
    return np.asarray(frames), timestamps


def test_counts_expected_blinks() -> None:
    """Шесть запланированных морганий должны быть найдены."""
    landmarks, timestamps = _make_sequence(n_frames=600, blink_every=100)
    result = analyze(landmarks, timestamps)

    assert result.blink_count == 6, f"Найдено {result.blink_count} морганий вместо 6"
    assert result.reliable


def test_relative_threshold_works_for_narrow_eyes() -> None:
    """Порог относительный, поэтому разрез глаз не должен влиять на счёт.

    Это главная причина, по которой порог не фиксированный: при абсолютном
    пороге узкие глаза считались бы закрытыми постоянно.
    """
    wide, timestamps = _make_sequence(open_height=16.0)
    narrow, _ = _make_sequence(open_height=6.0)

    assert analyze(wide, timestamps).blink_count == analyze(narrow, timestamps).blink_count


def test_rested_person_has_low_index() -> None:
    """Редкие короткие моргания без зевков — низкая усталость."""
    landmarks, timestamps = _make_sequence(blink_every=120, blink_frames=3)
    result = analyze(landmarks, timestamps)

    assert result.level is FatigueLevel.LOW
    assert result.index <= config.FATIGUE_LOW_MAX


def test_drowsy_person_scores_higher_than_rested() -> None:
    """Частые долгие моргания и зевок должны поднять индекс."""
    rested, timestamps = _make_sequence(blink_every=150, blink_frames=3)
    drowsy, _ = _make_sequence(blink_every=30, blink_frames=14, yawn_at=200)

    rested_result = analyze(rested, timestamps)
    drowsy_result = analyze(drowsy, timestamps)

    assert drowsy_result.index > rested_result.index
    assert drowsy_result.perclos > rested_result.perclos


def test_yawn_is_detected() -> None:
    """Долгое раскрытие рта считается зевком, короткое — нет."""
    with_yawn, timestamps = _make_sequence(yawn_at=200, yawn_frames=45)
    without_yawn, _ = _make_sequence(yawn_at=None)

    assert analyze(with_yawn, timestamps).yawn_count >= 1
    assert analyze(without_yawn, timestamps).yawn_count == 0


def test_short_speech_is_not_counted_as_yawn() -> None:
    """Кратко открытый рот — это речь, а не зевок."""
    landmarks, timestamps = _make_sequence(yawn_at=200, yawn_frames=8)
    assert analyze(landmarks, timestamps).yawn_count == 0


def test_short_recording_marked_unreliable() -> None:
    """Короткая запись помечается как ненадёжная, но не падает."""
    landmarks, timestamps = _make_sequence(n_frames=60)  # 2 секунды
    result = analyze(landmarks, timestamps)

    assert not result.reliable
    assert result.index == 0.0


def test_index_stays_in_range() -> None:
    """Индекс не должен вылезать за 0–100 даже при экстремальных входах."""
    landmarks, timestamps = _make_sequence(
        blink_every=10, blink_frames=8, yawn_at=100, yawn_frames=300
    )
    result = analyze(landmarks, timestamps)

    assert 0.0 <= result.index <= 100.0
