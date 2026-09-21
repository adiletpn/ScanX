"""Тесты реестра мошеннических номеров.

Реестр — мост между платформами: Android наполняет, iOS использует.
Проверяются три вещи, и вторая важнее остальных:

1. Один и тот же номер в разных написаниях — одна запись.
2. **Одного сигнала мало для предупреждения.** Мошенники подменяют номера,
   и по одному срабатыванию под подозрение попал бы человек,
   чей номер подставили.
3. Экспорт для iOS отсортирован — иначе система отвергнет весь список.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import config
from core import registry
from core.registry import NumberRecord, check, export_for_ios, normalize, report


@pytest.fixture(autouse=True)
def temp_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REGISTRY_FILE", tmp_path / "registry.json")
    yield


# ─────────────────────────────────────────────────────────────
# Нормализация
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "+7 707 123-45-67",
        "87071234567",
        "77071234567",
        "7071234567",
        "+7(707)123 45 67",
    ],
)
def test_all_spellings_collapse_to_one(raw: str) -> None:
    """Любое написание номера даёт один и тот же ключ.

    Без этого реестр наполнится дубликатами, и порог подтверждений
    никогда не наберётся: каждое написание считалось бы отдельным номером.
    """
    assert normalize(raw) == "+77071234567"


def test_empty_number_is_rejected() -> None:
    assert normalize("") == ""
    assert normalize("не номер") == ""
    assert report("абракадабра", ["sms_code"]) is None


# ─────────────────────────────────────────────────────────────
# Порог подтверждений
# ─────────────────────────────────────────────────────────────


def test_single_report_does_not_warn() -> None:
    """Одного срабатывания мало, чтобы обвинить номер."""
    report("+77071234567", ["sms_code"])
    assert check("+77071234567") is None


def test_second_report_confirms() -> None:
    """Повторное срабатывание подтверждает номер."""
    report("+77071234567", ["sms_code"])
    report("87071234567", ["urgency"])

    record = check("77071234567")
    assert record is not None
    assert record.reports == 2
    assert record.label


def test_schemes_accumulate_without_duplicates() -> None:
    """Схемы копятся, но не повторяются."""
    report("+77071234567", ["sms_code", "urgency"])
    report("+77071234567", ["urgency", "secrecy"])

    record = check("+77071234567")
    assert set(record.schemes) == {"sms_code", "urgency", "secrecy"}


def test_unknown_number_is_clean() -> None:
    """Незнакомый номер не вызывает подозрений."""
    assert check("+77079999999") is None


# ─────────────────────────────────────────────────────────────
# Срок давности
# ─────────────────────────────────────────────────────────────


def test_stale_record_is_ignored() -> None:
    """Устаревшая запись не предупреждает.

    Номера перепродают и переназначают. Вечный список со временем
    начинает врать на ни в чём не виноватых людях.
    """
    old = (datetime.now() - timedelta(days=config.REGISTRY_TTL_DAYS + 10)).isoformat()
    registry.save(
        {
            "+77071234567": NumberRecord(
                number="+77071234567",
                reports=5,
                first_seen=old,
                last_seen=old,
                schemes=["sms_code"],
            )
        }
    )
    assert check("+77071234567") is None


# ─────────────────────────────────────────────────────────────
# Экспорт для iOS
# ─────────────────────────────────────────────────────────────


def test_ios_export_is_sorted() -> None:
    """Список для Call Directory обязан идти по возрастанию.

    iOS молча отвергает неотсортированный список целиком — без ошибки
    и без подсказки, что именно не так.
    """
    for number in ("+77079999999", "+77071111111", "+77075555555"):
        report(number, ["sms_code"])
        report(number, ["urgency"])

    entries = export_for_ios()
    numbers = [e["number"] for e in entries]

    assert numbers == sorted(numbers)
    assert len(numbers) == 3
    assert all(isinstance(n, int) for n in numbers)


def test_ios_export_skips_unconfirmed() -> None:
    """Неподтверждённые номера в список не попадают."""
    report("+77071111111", ["sms_code"])          # одно сообщение
    report("+77072222222", ["sms_code"])
    report("+77072222222", ["urgency"])            # два сообщения

    numbers = [e["number"] for e in export_for_ios()]
    assert numbers == [77072222222]


def test_ios_export_has_label() -> None:
    """У каждой записи есть подпись для экрана вызова."""
    report("+77071234567", ["sms_code"])
    report("+77071234567", ["urgency"])

    entry = export_for_ios()[0]
    assert entry["label"]
    assert len(entry["label"]) <= 40, "длинную подпись iOS обрежет"


# ─────────────────────────────────────────────────────────────
# Устойчивость
# ─────────────────────────────────────────────────────────────


def test_corrupted_registry_does_not_crash() -> None:
    """Повреждённый файл не ломает защиту."""
    config.REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.REGISTRY_FILE.write_text("{сломано", encoding="utf-8")

    assert registry.load() == {}
    assert check("+77071234567") is None


def test_stats_are_consistent() -> None:
    """Сводка согласована с содержимым."""
    report("+77071111111", ["sms_code"])
    report("+77072222222", ["sms_code"])
    report("+77072222222", ["urgency"])

    s = registry.stats()
    assert s["total"] == 2
    assert s["confirmed"] == 1
    assert s["reports"] == 3
