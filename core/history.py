"""История замеров и личная норма.

Главная мысль модуля: **норма у каждого своя.** Справочные 60–100 ударов
описывают население, а не человека. У одного пульс покоя 54, у другого 88,
и оба здоровы. Сравнивать с усреднённой таблицей почти бесполезно —
она поднимет тревогу на первом и промолчит на втором.

Поэтому личная норма считается по его же прошлым замерам: медиана и разброс
за последние дни. Отклонением считается выход за пределы собственного
коридора, а не за границы справочника.

Хранение — обычный JSON рядом с приложением. База данных здесь была бы
лишней: замеров один-два в день, за год их наберётся несколько сотен.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import config

log = logging.getLogger(__name__)


@dataclass
class Measurement:
    """Один замер."""

    timestamp: str
    """Момент замера, ISO 8601."""

    bpm: float
    fatigue: float
    quality: str

    @property
    def moment(self) -> datetime:
        return datetime.fromisoformat(self.timestamp)


@dataclass(frozen=True)
class Baseline:
    """Личная норма, посчитанная по прошлым замерам."""

    bpm_median: float
    bpm_low: float
    """Нижняя граница личного коридора."""

    bpm_high: float
    """Верхняя граница личного коридора."""

    fatigue_median: float
    samples: int
    days: int

    @property
    def is_reliable(self) -> bool:
        """Хватает ли замеров, чтобы говорить о норме.

        По трём точкам коридор строить нельзя: он будет либо неправдоподобно
        узким, либо случайным. Пять — разумный минимум для демонстрации;
        в настоящем продукте стоило бы брать две недели.
        """
        return self.samples >= config.BASELINE_MIN_SAMPLES


@dataclass(frozen=True)
class Deviation:
    """Насколько замер отклонился от личной нормы."""

    has_baseline: bool
    is_deviation: bool
    bpm_delta: float
    """На сколько ударов отличается от медианы. Со знаком."""

    direction: str
    """«выше», «ниже» или «в норме»."""

    message: str
    """Готовая фраза для интерфейса и для уведомления детям."""

    should_notify: bool
    """Стоит ли сообщать близким."""


def _storage_path() -> Path:
    return config.HISTORY_FILE


def load() -> list[Measurement]:
    """Читает историю замеров.

    Повреждённый файл не должен ломать приложение: замеры важны,
    но не настолько, чтобы из-за них не открылся скан.
    """
    path = _storage_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [Measurement(**item) for item in raw]
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning("История повреждена, начинаю заново: %s", exc)
        return []


def save(measurements: list[Measurement]) -> None:
    """Записывает историю, оставляя только последние записи."""
    path = _storage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    trimmed = measurements[-config.HISTORY_MAX_ENTRIES :]
    path.write_text(
        json.dumps([asdict(m) for m in trimmed], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def add(bpm: float, fatigue: float, quality: str) -> list[Measurement]:
    """Добавляет замер в историю.

    Замеры плохого качества не сохраняются: личная норма, построенная
    на шуме, хуже отсутствия нормы — она будет тихо врать.
    """
    if quality == "poor":
        log.info("Замер плохого качества в историю не идёт")
        return load()

    history = load()
    history.append(
        Measurement(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            bpm=round(float(bpm), 1),
            fatigue=round(float(fatigue), 1),
            quality=quality,
        )
    )
    save(history)
    return history


def compute_baseline(history: list[Measurement] | None = None) -> Baseline:
    """Считает личную норму по последним дням.

    Медиана, а не среднее: один замер сразу после лестницы не должен
    сдвигать норму. По той же причине границы коридора берутся
    по процентилям, а не по стандартному отклонению.
    """
    history = load() if history is None else history

    cutoff = datetime.now() - timedelta(days=config.BASELINE_WINDOW_DAYS)
    recent = [m for m in history if m.moment >= cutoff]

    if not recent:
        return Baseline(0.0, 0.0, 0.0, 0.0, 0, config.BASELINE_WINDOW_DAYS)

    import numpy as np

    bpms = np.array([m.bpm for m in recent], dtype=float)
    fatigues = np.array([m.fatigue for m in recent], dtype=float)

    median = float(np.median(bpms))
    # Коридор по процентилям, но не уже минимальной ширины: при десяти
    # почти одинаковых замерах процентили сойдутся в точку, и тогда
    # любое естественное колебание выглядело бы отклонением.
    low = float(np.percentile(bpms, 15))
    high = float(np.percentile(bpms, 85))
    half_width = max((high - low) / 2.0, config.BASELINE_MIN_HALF_WIDTH)

    return Baseline(
        bpm_median=median,
        bpm_low=median - half_width,
        bpm_high=median + half_width,
        fatigue_median=float(np.median(fatigues)),
        samples=len(recent),
        days=config.BASELINE_WINDOW_DAYS,
    )


def check_deviation(bpm: float, baseline: Baseline | None = None) -> Deviation:
    """Сравнивает замер с личной нормой.

    Args:
        bpm: пульс текущего замера.
        baseline: личная норма; если не передана — считается заново.

    Returns:
        Deviation с готовой формулировкой для показа и для уведомления.
    """
    baseline = compute_baseline() if baseline is None else baseline

    if not baseline.is_reliable:
        left = config.BASELINE_MIN_SAMPLES - baseline.samples
        return Deviation(
            has_baseline=False,
            is_deviation=False,
            bpm_delta=0.0,
            direction="в норме",
            message=(
                f"Личная норма ещё набирается: нужно ещё {left} "
                f"{'замер' if left == 1 else 'замера'}."
            ),
            should_notify=False,
        )

    delta = bpm - baseline.bpm_median

    if bpm > baseline.bpm_high:
        direction = "выше"
    elif bpm < baseline.bpm_low:
        direction = "ниже"
    else:
        direction = "в норме"

    if direction == "в норме":
        return Deviation(
            has_baseline=True,
            is_deviation=False,
            bpm_delta=delta,
            direction=direction,
            message=(
                f"Пульс {bpm:.0f} — ваша обычная норма "
                f"{baseline.bpm_low:.0f}–{baseline.bpm_high:.0f}."
            ),
            should_notify=False,
        )

    # Уведомляем не на любом выходе за коридор, а на заметном:
    # иначе дети начнут получать сообщения через день и перестанут читать.
    sharp = abs(delta) >= config.DEVIATION_NOTIFY_BPM

    return Deviation(
        has_baseline=True,
        is_deviation=True,
        bpm_delta=delta,
        direction=direction,
        message=(
            f"Пульс {bpm:.0f} — это {direction} вашей обычной нормы "
            f"({baseline.bpm_low:.0f}–{baseline.bpm_high:.0f}), "
            f"на {abs(delta):.0f} уд/мин."
        ),
        should_notify=sharp,
    )


def recent_series(days: int | None = None) -> tuple[list[str], list[float]]:
    """Ряд замеров для графика: подписи дат и значения пульса."""
    days = config.BASELINE_WINDOW_DAYS if days is None else days
    cutoff = datetime.now() - timedelta(days=days)
    recent = [m for m in load() if m.moment >= cutoff]
    labels = [m.moment.strftime("%d.%m %H:%M") for m in recent]
    return labels, [m.bpm for m in recent]


def clear() -> None:
    """Стирает историю — нужно для демонстрации и для тестов."""
    path = _storage_path()
    if path.exists():
        path.unlink()
