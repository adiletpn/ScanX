"""Тесты личной нормы.

Главное требование: норма считается по ЭТОМУ человеку, а не по справочнику.
Пульс 54 у одного и 88 у другого — обе нормы, и система обязана понимать это
без подсказок.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import config
from core import history
from core.history import Measurement, check_deviation, compute_baseline


@pytest.fixture(autouse=True)
def temp_storage(tmp_path, monkeypatch):
    """Каждому тесту — своя история, чтобы они не влияли друг на друга."""
    monkeypatch.setattr(config, "HISTORY_FILE", tmp_path / "history.json")
    yield


def _series(values: list[float], days_back: int = 7) -> list[Measurement]:
    """Строит историю замеров, разложенную по последним дням."""
    now = datetime.now()
    step = timedelta(days=days_back / max(len(values), 1))
    return [
        Measurement(
            timestamp=(now - step * (len(values) - i)).isoformat(timespec="seconds"),
            bpm=v,
            fatigue=20.0,
            quality="good",
        )
        for i, v in enumerate(values)
    ]


# ─────────────────────────────────────────────────────────────
# Личная норма
# ─────────────────────────────────────────────────────────────


def test_baseline_needs_enough_samples() -> None:
    """По двум замерам норму не строим и честно об этом говорим."""
    history.save(_series([70.0, 72.0]))
    baseline = compute_baseline()

    assert not baseline.is_reliable
    deviation = check_deviation(95.0, baseline)
    assert not deviation.has_baseline
    assert "норма ещё набирается" in deviation.message.lower()


def test_low_resting_pulse_is_personal_norm() -> None:
    """Пульс 54 — норма для этого человека, хотя справочник начинается с 60."""
    history.save(_series([53.0, 55.0, 54.0, 52.0, 56.0, 54.0]))
    deviation = check_deviation(54.0)

    assert deviation.has_baseline
    assert not deviation.is_deviation
    assert deviation.direction == "в норме"


def test_high_resting_pulse_is_personal_norm() -> None:
    """Пульс 88 тоже может быть личной нормой."""
    history.save(_series([87.0, 89.0, 88.0, 90.0, 86.0, 88.0]))
    deviation = check_deviation(88.0)

    assert not deviation.is_deviation


def test_same_value_deviates_for_one_person_and_not_another() -> None:
    """Один и тот же пульс — отклонение у одного и норма у другого.

    Это и есть смысл личной нормы: справочная таблица так не умеет.
    """
    history.save(_series([53.0, 55.0, 54.0, 52.0, 56.0, 54.0]))
    calm_person = check_deviation(75.0)

    history.clear()
    history.save(_series([74.0, 76.0, 75.0, 73.0, 77.0, 75.0]))
    fast_person = check_deviation(75.0)

    assert calm_person.is_deviation
    assert not fast_person.is_deviation


def test_deviation_direction_is_named() -> None:
    """Отклонение описывается словом, а не только числом."""
    history.save(_series([60.0, 62.0, 61.0, 59.0, 63.0, 61.0]))

    assert check_deviation(95.0).direction == "выше"
    assert check_deviation(40.0).direction == "ниже"


def test_narrow_series_does_not_make_everything_a_deviation() -> None:
    """Серия почти одинаковых замеров не сужает коридор до точки.

    Без минимальной ширины коридора десять одинаковых замеров сделали бы
    отклонением любое естественное колебание на пару ударов.
    """
    history.save(_series([70.0] * 8))
    baseline = compute_baseline()

    assert baseline.bpm_high - baseline.bpm_low >= 2 * config.BASELINE_MIN_HALF_WIDTH
    assert not check_deviation(73.0).is_deviation


def test_median_resists_single_outlier() -> None:
    """Один замер после лестницы не сдвигает личную норму."""
    history.save(_series([70.0, 71.0, 69.0, 72.0, 70.0, 130.0]))
    baseline = compute_baseline()

    assert 68.0 <= baseline.bpm_median <= 73.0


# ─────────────────────────────────────────────────────────────
# Уведомления детям
# ─────────────────────────────────────────────────────────────


def test_small_deviation_does_not_notify() -> None:
    """Мелкое отклонение детям не шлём — иначе перестанут читать."""
    history.save(_series([70.0] * 6))
    deviation = check_deviation(78.0)
    assert not deviation.should_notify


def test_sharp_deviation_notifies() -> None:
    """Заметное отклонение — повод сообщить близким."""
    history.save(_series([70.0] * 6))
    deviation = check_deviation(100.0)

    assert deviation.is_deviation
    assert deviation.should_notify
    assert "выше" in deviation.message


# ─────────────────────────────────────────────────────────────
# Хранение
# ─────────────────────────────────────────────────────────────


def test_poor_quality_scan_is_not_stored() -> None:
    """Замер из шума не должен попадать в личную норму.

    Норма, построенная на плохих данных, хуже её отсутствия:
    она тихо врёт вместо того, чтобы честно молчать.
    """
    history.add(70.0, 20.0, "good")
    before = len(history.load())
    history.add(180.0, 90.0, "poor")

    assert len(history.load()) == before


def test_corrupted_file_does_not_crash() -> None:
    """Повреждённый файл истории не ломает приложение."""
    config.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.HISTORY_FILE.write_text("{это не json", encoding="utf-8")

    assert history.load() == []
    assert not compute_baseline().is_reliable


def test_history_is_trimmed() -> None:
    """История не растёт бесконечно."""
    history.save(_series([70.0] * 10, days_back=3))
    assert len(history.load()) <= config.HISTORY_MAX_ENTRIES


def test_series_for_chart_matches_history() -> None:
    """Ряд для графика согласован с историей."""
    history.save(_series([70.0, 72.0, 68.0, 71.0, 69.0], days_back=5))
    labels, values = history.recent_series()

    assert len(labels) == len(values) == 5
