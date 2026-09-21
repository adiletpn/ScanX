"""Тесты оценки пульса на синтетических сигналах.

Синтетика — единственный способ проверить rPPG объективно: здесь мы знаем
истинную частоту и можем сравнить с ней. На реальном видео эталона нет.

Проверяем три вещи:
    1. На чистой модели пульса частота определяется точно.
    2. Неравномерные метки времени (плавающий FPS телефона) не ломают оценку.
    3. Короткий сигнал даёт понятную ошибку, а не падение.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from core.rppg import (
    HeartRateResult,
    SignalTooShortError,
    bandpass,
    estimate_heart_rate,
    pos_projection,
    resample_uniform,
)

#: Истинная частота синтетического пульса: 1.2 Гц = 72 уд/мин.
TRUE_HZ = 1.2
TRUE_BPM = TRUE_HZ * 60.0

#: Допуск из ТЗ.
TOLERANCE_BPM = 2.0


def _make_pulse(
    duration_s: float = 30.0,
    fps: float = 30.0,
    freq_hz: float = TRUE_HZ,
    noise: float = 0.3,
    trend: float = 2.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Синусоида нужной частоты плюс шум и линейный тренд.

    Тренд имитирует медленный дрейф освещения и сползание ROI,
    шум — сенсорный шум камеры. И то и другое должно сниматься
    детрендингом и полосовым фильтром.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, 1.0 / fps)
    pulse = np.sin(2.0 * np.pi * freq_hz * t)
    drift = trend * t / duration_s
    return t, pulse + drift + noise * rng.standard_normal(t.size)


def _make_rgb_pulse(
    duration_s: float = 30.0,
    fps: float = 30.0,
    freq_hz: float = TRUE_HZ,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Правдоподобный RGB-ряд кожи с пульсовой модуляцией.

    Амплитуда пульса взята 1% от уровня сигнала — примерно столько и даёт
    реальный кровоток. Зелёный модулируется сильнее: гемоглобин поглощает
    его лучше всего, на этом и держится метод.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, 1.0 / fps)
    pulse = np.sin(2.0 * np.pi * freq_hz * t)

    # Медленное колебание освещения — главная помеха, которую давит POS.
    illumination = 1.0 + 0.05 * np.sin(2.0 * np.pi * 0.08 * t)

    base = np.array([180.0, 140.0, 120.0])  # типичный тон кожи
    gains = np.array([0.006, 0.012, 0.004])  # G модулируется сильнее

    rgb = base[None, :] * illumination[:, None]
    rgb = rgb * (1.0 + gains[None, :] * pulse[:, None])
    rgb += rng.standard_normal(rgb.shape) * 0.4

    return t, rgb


# ─────────────────────────────────────────────────────────────
# 1. Точность на чистой синтетике
# ─────────────────────────────────────────────────────────────


def test_pulse_72_bpm_from_clean_signal() -> None:
    """Синусоида 1.2 Гц с шумом и трендом → 72 ± 2 уд/мин."""
    t, x = _make_pulse()
    result = estimate_heart_rate(t, x)

    assert isinstance(result, HeartRateResult)
    assert abs(result.bpm - TRUE_BPM) <= TOLERANCE_BPM, (
        f"Ожидали {TRUE_BPM} ± {TOLERANCE_BPM}, получили {result.bpm:.2f}"
    )


def test_pos_recovers_pulse_from_rgb() -> None:
    """POS вытаскивает пульс из RGB, несмотря на дрейф освещения."""
    t, rgb = _make_rgb_pulse()
    result = estimate_heart_rate(t, rgb, method="pos")

    assert abs(result.bpm - TRUE_BPM) <= TOLERANCE_BPM, (
        f"POS дал {result.bpm:.2f} вместо {TRUE_BPM}"
    )
    # Чистая синтетика обязана давать высокий SNR. Если он просел —
    # что-то сломалось в конвейере, даже когда частота случайно сошлась.
    assert result.snr_db > config.QUALITY_MIN_SNR_OK


def test_spread_is_small_on_stationary_signal() -> None:
    """У стационарного сигнала разброс по окнам должен быть маленьким."""
    t, x = _make_pulse(noise=0.2)
    result = estimate_heart_rate(t, x)

    assert result.n_windows > 5
    assert result.bpm_spread < 5.0, f"Разброс {result.bpm_spread:.2f} слишком велик"


@pytest.mark.parametrize("bpm", [50.0, 72.0, 95.0, 130.0])
def test_various_heart_rates(bpm: float) -> None:
    """Метод работает во всём физиологичном диапазоне, не только на 72."""
    t, x = _make_pulse(freq_hz=bpm / 60.0)
    result = estimate_heart_rate(t, x)
    assert abs(result.bpm - bpm) <= TOLERANCE_BPM


# ─────────────────────────────────────────────────────────────
# 2. Неравномерные метки времени
# ─────────────────────────────────────────────────────────────


def test_irregular_timestamps_do_not_break_estimate() -> None:
    """Плавающий FPS не должен сдвигать оценку.

    Телефон подстраивает выдержку под освещение, и интервалы между кадрами
    гуляют. Раз мы интерполируем по реальному времени, результат обязан
    совпасть с равномерным случаем.
    """
    rng = np.random.default_rng(11)
    t, x = _make_pulse(duration_s=30.0)

    # Дрожание меток ±30% от интервала плюс выпадающие кадры.
    jitter = rng.uniform(-0.3, 0.3, t.size) / 30.0
    t_irregular = np.sort(t + jitter)
    keep = rng.random(t.size) > 0.1  # теряем ~10% кадров
    t_irregular, x_irregular = t_irregular[keep], x[keep]

    result = estimate_heart_rate(t_irregular, x_irregular)
    assert abs(result.bpm - TRUE_BPM) <= TOLERANCE_BPM + 1.0


def test_unsorted_and_duplicate_timestamps_are_cleaned() -> None:
    """Перемешанные и повторяющиеся метки чинятся, а не роняют модуль."""
    t, x = _make_pulse(duration_s=25.0)

    t_dirty = np.concatenate([t, t[:50]])  # дубликаты
    x_dirty = np.concatenate([x, x[:50]])
    order = np.random.default_rng(3).permutation(t_dirty.size)  # перемешали

    grid, values = resample_uniform(t_dirty[order], x_dirty[order])

    assert np.all(np.diff(grid) > 0), "Сетка должна строго возрастать"
    assert np.isfinite(values).all()


def test_resample_matches_known_function() -> None:
    """Интерполяция не искажает значения: проверяем на прямой."""
    t = np.array([0.0, 0.5, 1.7, 3.0, 12.5, 20.0])
    values = 2.0 * t + 1.0  # y = 2x + 1

    grid, out = resample_uniform(t, values, fps=10.0)

    expected = 2.0 * grid + 1.0
    assert np.allclose(out[:, 0], expected, atol=1e-9)


# ─────────────────────────────────────────────────────────────
# 3. Слишком короткий сигнал — понятная ошибка
# ─────────────────────────────────────────────────────────────


def test_short_recording_raises_clear_error() -> None:
    """Запись короче минимума → SignalTooShortError, не крэш."""
    t, x = _make_pulse(duration_s=5.0)

    with pytest.raises(SignalTooShortError) as exc:
        estimate_heart_rate(t, x)

    # Сообщение уходит прямо пользователю, поэтому оно должно быть про суть.
    assert "с" in str(exc.value)


def test_single_frame_raises() -> None:
    """Один кадр — тоже понятная ошибка."""
    with pytest.raises(SignalTooShortError):
        estimate_heart_rate([0.0], [1.0])


def test_bandpass_rejects_too_short_signal() -> None:
    """Фильтр не должен падать с невнятным ValueError из scipy."""
    with pytest.raises(SignalTooShortError):
        bandpass(np.zeros(20))


def test_mismatched_lengths_raise_value_error() -> None:
    """Расхождение длин меток и значений — ошибка программиста, не данных."""
    with pytest.raises(ValueError):
        resample_uniform([0.0, 1.0, 2.0], [1.0, 2.0])


def test_pos_rejects_wrong_shape() -> None:
    """POS принимает только (N, 3)."""
    with pytest.raises(ValueError):
        pos_projection(np.zeros((100, 2)))


# ─────────────────────────────────────────────────────────────
# Целостность результата
# ─────────────────────────────────────────────────────────────


def test_result_arrays_are_consistent() -> None:
    """Волна, метки и спектр должны быть согласованы — их рисует UI."""
    t, rgb = _make_rgb_pulse()
    result = estimate_heart_rate(t, rgb)

    assert result.wave.size == result.wave_times.size
    assert result.freqs.size == result.spectrum.size
    assert np.isfinite(result.wave).all()
    assert 0.0 <= result.spectrum.max() <= 1.0 + 1e-9
    assert config.HR_MIN_BPM <= result.bpm <= config.HR_MAX_BPM
