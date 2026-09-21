"""Реестр мошеннических номеров.

Мост между платформами. На Android приложение разбирает разговор и, поймав
схему, записывает номер сюда. На iOS разобрать разговор невозможно —
Apple не даёт микрофон во время звонка, — но можно предупредить ДО ответа,
показав подпись прямо на экране входящего вызова через Call Directory.

    Android: слышит разговор ──▶ номер в реестр ──▶ iOS: предупреждает до ответа

Получается сеть: чем больше пользователей на Android, тем лучше защищены
владельцы iPhone. Слабость платформы превращается в архитектуру.

Два свойства, которые стоит держать в голове:

* **Проверка номера работает офлайн.** Список лежит на устройстве,
  как база антивируса. Интернет нужен только чтобы список пополнить.
* **Один сигнал номер не осуждает.** Мошенники подменяют номера, и под
  раздачу легко попадёт обычный человек. Поэтому есть порог подтверждений
  и срок давности.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import config

log = logging.getLogger(__name__)


@dataclass
class NumberRecord:
    """Что мы знаем об одном номере."""

    number: str
    """Номер в нормализованном виде: только цифры с ведущим плюсом."""

    reports: int
    """Сколько раз на нём срабатывал движок."""

    first_seen: str
    last_seen: str
    schemes: list[str] = field(default_factory=list)
    """Коды схем, которые на нём встречались."""

    @property
    def last_moment(self) -> datetime:
        return datetime.fromisoformat(self.last_seen)

    @property
    def is_confirmed(self) -> bool:
        """Достаточно ли подтверждений, чтобы предупреждать о номере.

        Одного срабатывания мало: мошенники подменяют номера, и тогда
        предупреждение получил бы ни в чём не виноватый человек,
        чей номер подставили.
        """
        return self.reports >= config.REGISTRY_MIN_REPORTS

    @property
    def is_fresh(self) -> bool:
        """Не устарела ли запись.

        Номера перепродаются и переназначаются. Держать их вечно значит
        со временем накопить список, который врёт.
        """
        return datetime.now() - self.last_moment <= timedelta(
            days=config.REGISTRY_TTL_DAYS
        )

    @property
    def label(self) -> str:
        """Подпись, которую покажет система на экране входящего звонка.

        Длина ограничена: iOS обрезает длинные подписи, а человеку
        в момент звонка нужно одно слово, а не абзац.
        """
        return "Возможно, мошенник"


def normalize(number: str) -> str:
    """Приводит номер к единому виду.

    Один и тот же номер приходит как «+7 707 123-45-67», «87071234567»
    и «77071234567». Без нормализации реестр наполнится дубликатами,
    и порог подтверждений никогда не наберётся.
    """
    digits = re.sub(r"\D", "", number)
    if not digits:
        return ""

    # Казахстан и Россия: восьмёрка в начале — это та же семёрка.
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    if len(digits) == 10:
        digits = "7" + digits

    return "+" + digits


def load() -> dict[str, NumberRecord]:
    """Читает реестр. Повреждённый файл не должен ломать защиту."""
    path = config.REGISTRY_FILE
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {k: NumberRecord(**v) for k, v in raw.items()}
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning("Реестр повреждён, начинаю заново: %s", exc)
        return {}


def save(records: dict[str, NumberRecord]) -> None:
    """Записывает реестр, выбрасывая устаревшие записи."""
    path = config.REGISTRY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    alive = {k: v for k, v in records.items() if v.is_fresh}
    path.write_text(
        json.dumps({k: asdict(v) for k, v in alive.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def report(number: str, schemes: list[str]) -> NumberRecord | None:
    """Отмечает номер как замеченный в мошеннической схеме.

    Args:
        number: номер звонившего в любом виде.
        schemes: коды схем, сработавших в разговоре.

    Returns:
        Запись о номере, либо None, если номер не разобрать.
    """
    key = normalize(number)
    if not key:
        log.warning("Не разобрал номер: %r", number)
        return None

    now = datetime.now().isoformat(timespec="seconds")
    records = load()

    existing = records.get(key)
    if existing is None:
        records[key] = NumberRecord(
            number=key, reports=1, first_seen=now, last_seen=now, schemes=list(schemes)
        )
    else:
        existing.reports += 1
        existing.last_seen = now
        for code in schemes:
            if code not in existing.schemes:
                existing.schemes.append(code)

    save(records)
    log.info("Номер %s: %d сообщений", key, records[key].reports)
    return records[key]


def check(number: str) -> NumberRecord | None:
    """Проверяет номер по реестру. Это и есть работа iOS-расширения.

    Returns:
        Запись, если номер подтверждён и не устарел; иначе None.
    """
    key = normalize(number)
    if not key:
        return None
    record = load().get(key)
    if record is None or not record.is_confirmed or not record.is_fresh:
        return None
    return record


def export_for_ios() -> list[dict[str, object]]:
    """Готовит список для iOS Call Directory Extension.

    Расширение требует номера **в виде целых чисел и строго по возрастанию** —
    иначе система молча отвергнет весь список. Сортировка здесь, а не
    на устройстве: ошибиться в ней легко, а диагностировать нечем.

    Returns:
        Список записей, отсортированный по номеру.
    """
    confirmed = [r for r in load().values() if r.is_confirmed and r.is_fresh]

    entries = []
    for record in confirmed:
        digits = record.number.lstrip("+")
        if not digits.isdigit():
            continue
        entries.append({"number": int(digits), "label": record.label})

    entries.sort(key=lambda e: e["number"])
    log.info("Для iOS подготовлено %d номеров", len(entries))
    return entries


def stats() -> dict[str, int]:
    """Сводка по реестру — для интерфейса."""
    records = load()
    return {
        "total": len(records),
        "confirmed": sum(1 for r in records.values() if r.is_confirmed and r.is_fresh),
        "reports": sum(r.reports for r in records.values()),
    }


def clear() -> None:
    """Очищает реестр — для демонстрации и тестов."""
    if config.REGISTRY_FILE.exists():
        config.REGISTRY_FILE.unlink()
