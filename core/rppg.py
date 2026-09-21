"""Оценка пульса из RGB-рядов лица (rPPG).

Идея метода: с каждым ударом сердца меняется кровенаполнение кожи, и камера
ловит микроколебания её цвета — слишком слабые для глаза, но различимые
статистически. Задача модуля — вытащить этот сигнал из шума и найти его частоту.

Конвейер:
    RGB по кадрам (неравномерные метки времени, видео с телефона имеет
    плавающий FPS)
    → интерполяция на равномерную сетку 30 Гц
    → POS (Wang et al., 2017): проекция, устойчивая к изменению освещения
    → детрендинг, z-нормализация, полосовой фильтр 0.75–3.0 Гц
    → спектр в скользящих окнах по 10 с
    → медиана частот по окнам = пульс, межквартильный размах = разброс

Литература:
    Wang et al. «Algorithmic Principles of Remote PPG», IEEE TBME, 2017.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import signal as sps

import config

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]


class SignalTooShortError(ValueError):
    """Сигнала не хватает для устойчивой оценки.

    Отдельный тип, потому что это не баг, а нормальная ситуация: пользователю
    нужно показать «запишите видео подлиннее», а не сообщение об ошибке.
    """


@dataclass(frozen=True)
class HeartRateResult:
    """Результат оценки пульса."""

    bpm: float
    """Итоговый пульс — медиана по скользящим окнам, уд/мин."""

    bpm_spread: float
    """Межквартильный размах оценок по окнам. Большой разброс = сигнал шумный."""

    snr_db: float
    """Отношение сигнал/шум спектра. Ниже ~1.5 дБ доверять числу нельзя."""

    peak_freq_hz: float
    """Частота основного пика спектра."""

    n_windows: int
    """Сколько окон участвовало в оценке."""

    window_bpms: FloatArray = field(repr=False)
    """Оценки по каждому окну — для диагностики."""

    wave: FloatArray = field(repr=False)
    """Отфильтрованная пульсовая волна (для графика)."""

    wave_times: FloatArray = field(repr=False)
    """Метки времени волны, секунды от начала записи."""

    freqs: FloatArray = field(repr=False)
    """Сетка частот спектра, Гц."""

    spectrum: FloatArray = field(repr=False)
    """Амплитудный спектр, нормированный на максимум."""


# ─────────────────────────────────────────────────────────────
# Подготовка сигнала
# ─────────────────────────────────────────────────────────────


def resample_uniform(
    timestamps_s: npt.ArrayLike,
    values: npt.ArrayLike,
    fps: float = config.RESAMPLE_FPS,
) -> tuple[FloatArray, FloatArray]:
    """Приводит ряд с неравномерными метками времени к равномерной сетке.

    Видео с телефона почти никогда не имеет стабильного FPS: камера
    подстраивает выдержку под освещение, и интервалы между кадрами плавают.
    Если считать спектр по номерам кадров, частота «поедет». Поэтому работаем
    с реальным временем и интерполируем на сетку с постоянным шагом.

    Args:
        timestamps_s: метки времени кадров в секундах, форма (N,).
        values: значения, форма (N,) или (N, C) для нескольких каналов.
        fps: частота итоговой сетки, Гц.

    Returns:
        Кортеж (равномерные метки времени, интерполированные значения).

    Raises:
        SignalTooShortError: если после очистки осталось меньше двух точек
            или запись короче минимальной длительности.
    """
    ts = np.asarray(timestamps_s, dtype=np.float64).ravel()
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]

    if ts.shape[0] != vals.shape[0]:
        raise ValueError(
            f"Длины не совпадают: {ts.shape[0]} меток и {vals.shape[0]} значений"
        )
    if ts.size < 2:
        raise SignalTooShortError("Слишком мало кадров для анализа")

    # Метки времени должны строго возрастать. В реальных видео встречаются
    # и повторы, и перестановки — сортируем и выкидываем дубликаты,
    # иначе np.interp вернёт мусор без всякого предупреждения.
    order = np.argsort(ts, kind="stable")
    ts, vals = ts[order], vals[order]
    keep = np.concatenate(([True], np.diff(ts) > 1e-9))
    ts, vals = ts[keep], vals[keep]

    if ts.size < 2:
        raise SignalTooShortError("После очистки меток времени не осталось данных")

    duration = float(ts[-1] - ts[0])
    if duration < config.MIN_DURATION_SEC:
        raise SignalTooShortError(
            f"Длительность записи {duration:.1f} с — нужно хотя бы "
            f"{config.MIN_DURATION_SEC:.0f} с"
        )

    n_out = int(np.floor(duration * fps)) + 1
    grid = ts[0] + np.arange(n_out, dtype=np.float64) / fps

    out = np.empty((n_out, vals.shape[1]), dtype=np.float64)
    for ch in range(vals.shape[1]):
        out[:, ch] = np.interp(grid, ts, vals[:, ch])

    return grid - grid[0], out


def pos_projection(rgb: FloatArray, fps: float = config.RESAMPLE_FPS) -> FloatArray:
    """Извлекает пульсовой сигнал методом POS (Plane-Orthogonal-to-Skin).

    POS проецирует RGB на плоскость, ортогональную направлению изменения
    яркости кожи. За счёт этого колебания освещения — главный источник помех
    при съёмке на телефон — подавляются, а пульсовая компонента остаётся.

    Args:
        rgb: массив (N, 3) средних R, G, B на равномерной сетке.
        fps: частота сетки, Гц.

    Returns:
        Одномерный пульсовой сигнал длины N.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"Ожидается массив (N, 3), получен {rgb.shape}")

    n = rgb.shape[0]
    # Окно 1.6 с — из оригинальной статьи: примерно два сердечных цикла.
    win = max(int(round(1.6 * fps)), 2)
    if n < win:
        raise SignalTooShortError("Сигнал короче одного окна POS")

    out = np.zeros(n, dtype=np.float64)

    # Матрица проекции из статьи. Первая строка убирает общую яркость,
    # вторая выделяет компоненту, ортогональную ей.
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])

    for start in range(0, n - win + 1):
        block = rgb[start : start + win]

        # Временная нормировка: делим на среднее по окну. Так уходит
        # зависимость от тона кожи и общего уровня освещённости.
        mean = block.mean(axis=0)
        mean[mean == 0] = 1e-9
        normalized = block / mean

        s = projection @ normalized.T  # (2, win)

        # Alpha-тюнинг: подмешиваем вторую компоненту так, чтобы
        # погасить остаточную пульсацию яркости.
        std1, std2 = s[0].std(), s[1].std()
        alpha = std1 / std2 if std2 > 1e-12 else 0.0
        h = s[0] + alpha * s[1]
        h -= h.mean()

        # Overlap-add: окна перекрываются, вклады суммируются.
        out[start : start + win] += h

    return out


def green_projection(rgb: FloatArray) -> FloatArray:
    """Запасной метод: только зелёный канал.

    Гемоглобин сильнее всего поглощает зелёный, поэтому канал G несёт
    основной пульсовой сигнал. Проще POS, но заметно хуже переносит движение
    и смену освещения. Включается через ``config.RPPG_METHOD``.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    green = rgb[:, 1]
    return green - green.mean()


def bandpass(
    x: FloatArray,
    fps: float = config.RESAMPLE_FPS,
    band_hz: tuple[float, float] = config.HR_BAND_HZ,
    order: int = config.FILTER_ORDER,
) -> FloatArray:
    """Полосовой фильтр Баттерворта с нулевым фазовым сдвигом.

    Реализован через second-order sections: на 4-м порядке прямая форма
    уже теряет устойчивость из-за накопления ошибок округления.
    ``sosfiltfilt`` проходит сигнал дважды, вперёд и назад, поэтому
    фаза не смещается — важно, раз мы потом ищем пики волны.

    Raises:
        SignalTooShortError: если сигнал короче, чем нужно фильтру для
            краевого дополнения (``padlen``).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    nyquist = fps / 2.0
    low, high = band_hz[0] / nyquist, band_hz[1] / nyquist
    if not 0.0 < low < high < 1.0:
        raise ValueError(f"Некорректная полоса {band_hz} при fps={fps}")

    sos = sps.butter(order, [low, high], btype="bandpass", output="sos")

    # sosfiltfilt дополняет сигнал с краёв отражением. Если длины не хватает,
    # scipy бросает невнятный ValueError — перехватываем заранее.
    padlen = 3 * (2 * sos.shape[0] + 1)
    if x.size <= padlen:
        raise SignalTooShortError(
            f"Сигнал из {x.size} отсчётов короче минимума для фильтра ({padlen + 1})"
        )

    return np.asarray(sps.sosfiltfilt(sos, x), dtype=np.float64)


def preprocess(x: FloatArray, fps: float = config.RESAMPLE_FPS) -> FloatArray:
    """Детрендинг → z-нормализация → полосовой фильтр."""
    x = np.asarray(x, dtype=np.float64).ravel()

    # Линейный тренд возникает из-за медленного дрейфа освещения и
    # сползания ROI. Для спектра это низкочастотный мусор.
    x = sps.detrend(x, type="linear")

    std = x.std()
    x = x / std if std > 1e-12 else x

    return bandpass(x, fps=fps)


# ─────────────────────────────────────────────────────────────
# Спектральная оценка частоты
# ─────────────────────────────────────────────────────────────


def _parabolic_peak(magnitude: FloatArray, k: int) -> float:
    """Уточняет положение пика параболой по трём точкам.

    Разрешение БПФ ограничено длиной окна: на 10 с это 0.1 Гц, то есть
    целых 6 уд/мин. Подгонка параболы по вершине и двум соседям даёт
    дробный индекс и снимает эту ступеньку.

    Returns:
        Поправка к индексу ``k`` в пределах ±0.5.
    """
    if k <= 0 or k >= magnitude.size - 1:
        return 0.0
    a, b, c = magnitude[k - 1], magnitude[k], magnitude[k + 1]
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-15:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))


def _spectrum(
    x: FloatArray, fps: float, pad_factor: int = config.FFT_PAD_FACTOR
) -> tuple[FloatArray, FloatArray]:
    """Амплитудный спектр с окном Ханна и дополнением нулями.

    Окно Ханна гасит разрывы на краях отрезка — без него в спектре
    появляются ложные боковые лепестки. Нули не добавляют информации,
    но сгущают сетку частот, и пик становится видно точнее.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    windowed = x * np.hanning(x.size)
    n_fft = int(2 ** np.ceil(np.log2(max(x.size * pad_factor, 2))))
    spec = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)
    return freqs, spec


def _peak_frequency(freqs: FloatArray, spec: FloatArray) -> float:
    """Ищет пик спектра внутри физиологичной полосы пульса."""
    lo, hi = config.HR_MIN_BPM / 60.0, config.HR_MAX_BPM / 60.0
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        raise SignalTooShortError("Полоса пульса не попала в спектр")

    idx = np.flatnonzero(band)
    local_peak = int(idx[np.argmax(spec[idx])])
    offset = _parabolic_peak(spec, local_peak)

    df = float(freqs[1] - freqs[0])
    return float(freqs[local_peak] + offset * df)


def _snr_db(freqs: FloatArray, spec: FloatArray, peak_hz: float) -> float:
    """Отношение сигнал/шум спектра, дБ.

    Сигналом считаем энергию вокруг основной частоты и её второй гармоники
    (сердечный ритм не синусоида, часть энергии всегда уходит наверх),
    шумом — остальное внутри полосы пульса.
    """
    lo, hi = config.HR_MIN_BPM / 60.0, config.HR_MAX_BPM / 60.0
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return 0.0

    power = spec**2
    tol = 0.2  # Гц — ширина «окна гармоники»
    harmonics = (np.abs(freqs - peak_hz) <= tol) | (
        np.abs(freqs - 2.0 * peak_hz) <= tol
    )

    signal_power = float(power[band & harmonics].sum())
    noise_power = float(power[band & ~harmonics].sum())

    if noise_power <= 1e-15:
        return 30.0  # практический потолок, дальше число смысла не несёт
    if signal_power <= 1e-15:
        return 0.0

    return float(10.0 * np.log10(signal_power / noise_power))


# ─────────────────────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────────────────────


def estimate_heart_rate(
    timestamps_s: npt.ArrayLike,
    rgb: npt.ArrayLike,
    fps: float = config.RESAMPLE_FPS,
    method: str | None = None,
) -> HeartRateResult:
    """Оценивает пульс по ряду средних RGB лица.

    Оценка ведётся по скользящим окнам, а не по всей записи сразу.
    Итог — медиана частот по окнам: одно испорченное движением окно
    её почти не сдвигает, тогда как единый спектр оно бы перекосило.
    Разброс между окнами сам по себе полезен — он показывает,
    насколько устойчив результат.

    Args:
        timestamps_s: метки времени кадров, секунды.
        rgb: массив (N, 3) средних R, G, B по ROI, либо (N,) — готовый сигнал.
        fps: частота равномерной сетки, Гц.
        method: ``"pos"`` или ``"green"``; по умолчанию из config.

    Returns:
        HeartRateResult с пульсом, разбросом, SNR, волной и спектром.

    Raises:
        SignalTooShortError: запись слишком короткая для оценки.
    """
    method = method or config.RPPG_METHOD

    grid_t, resampled = resample_uniform(timestamps_s, rgb, fps=fps)

    if resampled.shape[1] == 3:
        raw = (
            pos_projection(resampled, fps=fps)
            if method == "pos"
            else green_projection(resampled)
        )
    elif resampled.shape[1] == 1:
        raw = resampled[:, 0]  # уже готовый одномерный сигнал
    else:
        raise ValueError(f"Ожидается 1 или 3 канала, получено {resampled.shape[1]}")

    wave = preprocess(raw, fps=fps)

    win_len = int(round(config.HR_WINDOW_SEC * fps))
    step = max(int(round(config.HR_STEP_SEC * fps)), 1)

    # Если на полное окно не хватает, считаем по всей записи целиком:
    # оценка будет грубее, но лучше, чем отказ.
    if wave.size < win_len:
        log.info(
            "Запись короче окна %.0f с — оцениваем по всему сигналу",
            config.HR_WINDOW_SEC,
        )
        starts = [0]
        win_len = wave.size
    else:
        starts = list(range(0, wave.size - win_len + 1, step))

    window_freqs: list[float] = []
    for start in starts:
        chunk = wave[start : start + win_len]
        freqs_w, spec_w = _spectrum(chunk, fps)
        try:
            window_freqs.append(_peak_frequency(freqs_w, spec_w))
        except SignalTooShortError:
            continue

    if not window_freqs:
        raise SignalTooShortError("Не удалось оценить частоту ни в одном окне")

    freqs_arr = np.asarray(window_freqs, dtype=np.float64)
    bpms = freqs_arr * 60.0

    bpm = float(np.median(bpms))
    spread = float(np.percentile(bpms, 75) - np.percentile(bpms, 25))

    # Спектр по всей записи — он ровнее оконного и идёт на график.
    freqs_full, spec_full = _spectrum(wave, fps)
    peak_hz = bpm / 60.0
    snr = _snr_db(freqs_full, spec_full, peak_hz)

    spec_max = spec_full.max()
    spec_norm = spec_full / spec_max if spec_max > 1e-15 else spec_full

    log.info(
        "Пульс %.1f уд/мин (разброс %.1f, SNR %.1f дБ, окон %d)",
        bpm, spread, snr, len(window_freqs),
    )

    return HeartRateResult(
        bpm=bpm,
        bpm_spread=spread,
        snr_db=snr,
        peak_freq_hz=peak_hz,
        n_windows=len(window_freqs),
        window_bpms=bpms,
        wave=wave,
        wave_times=grid_t,
        freqs=freqs_full,
        spectrum=spec_norm,
    )
