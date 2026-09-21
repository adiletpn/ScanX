"""Консилиум: разбор случая несколькими специалистами сразу.

Человек описывает жалобу один раз. Система определяет, кого из врачей
это касается, каждый даёт короткое мнение со своей стороны, и в конце
выводится общее заключение.

Смысл не в количестве мнений, а в том, что разные специальности смотрят
на одну жалобу под разными углами. Головная боль с давлением 150/95 —
для кардиолога это вопрос гипертонии, для невролога характер боли,
для эндокринолога возможная вторичная причина у молодого пациента.
Очно человек получил бы это за три визита и три недели.

Конвейер:
    жалоба → выбор специальностей → мнение каждого → сводное заключение
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import config
from ai import ollama_client, prompts
from ai.specialties import SPECIALTIES, Specialty, get as get_specialty

log = logging.getLogger(__name__)

#: Сколько специалистов собираем. Три — компромисс: два дают слишком узкую
#: картину, от четырёх и больше ответы начинают повторяться, а ожидание
#: на локальной модели становится некомфортным.
COUNCIL_SIZE = 3


@dataclass
class Opinion:
    """Мнение одного специалиста."""

    specialty: Specialty
    text: str


SYSTEM_ROUTER = f"""Ты — регистратура клиники {config.APP_NAME}.
По жалобе пациента ты решаешь, каким специалистам её показать.

Доступные специальности (используй ТОЛЬКО их, ровно в этом написании):
{", ".join(s.name for s in SPECIALTIES)}

Ответь ОДНОЙ строкой: {COUNCIL_SIZE} названия через запятую, самое подходящее
первым. Никаких пояснений, никакого текста до или после.

Пример ответа:
Невролог, Кардиолог, Терапевт
"""

SYSTEM_SUMMARY = f"""Ты — председатель консилиума в приложении {config.APP_NAME}.
Перед тобой мнения нескольких специалистов по одному случаю.
Сведи их в короткое общее заключение.

## Формат — ровно эти пять строк

СОГЛАСИЕ: в чём мнения сходятся, одно предложение
РАЗНОГЛАСИЯ: в чём расходятся, либо «нет»
СРОЧНОСТЬ: одно слово — СРОЧНО / ПЛАНОВО / САМОСТОЯТЕЛЬНО
ГЛАВНЫЙ ВРАЧ: одна специальность, к кому идти в первую очередь
ПЛАН: 3 конкретных пункта через «;» — что сделать в ближайшие дни

Никакого текста до или после этих строк.
"""


def _parse_specialties(raw: str) -> list[Specialty]:
    """Разбирает ответ маршрутизатора в список специальностей.

    Модель на 4B легко добавляет вступление или пишет названия в падеже,
    поэтому ищем совпадения по всему тексту, а не разбираем строку строго.
    Если распознать не удалось — собираем консилиум по умолчанию,
    а не падаем: пустое заключение хуже приблизительного.
    """
    lowered = raw.lower()

    # Порядок важен: модель ставит самого подходящего врача первым,
    # и это нужно сохранить. Поэтому сортируем по позиции вхождения
    # в тексте, а не по порядку объявления в реестре.
    hits: list[tuple[int, Specialty]] = []
    for specialty in SPECIALTIES:
        # Сравниваем по корню: «неврологу», «врач-невролог», «Невролог».
        position = lowered.find(specialty.name.lower()[:6])
        if position >= 0:
            hits.append((position, specialty))

    found = [specialty for _, specialty in sorted(hits, key=lambda x: x[0])]

    if not found:
        log.warning("Маршрутизатор не вернул специальностей: %r", raw[:120])

    # Добираем состав запасными: терапевт смотрит на случай целиком,
    # невролог и кардиолог покрывают самые частые обращения.
    for key in ("therapist", "neurologist", "cardiologist"):
        if len(found) >= COUNCIL_SIZE:
            break
        fallback = get_specialty(key)
        if fallback not in found:
            found.append(fallback)

    return found[:COUNCIL_SIZE]


def choose_specialists(complaint: str, scan_context: str = "") -> list[Specialty]:
    """Определяет, каким специалистам показать жалобу.

    Raises:
        ollama_client.OllamaError: модель недоступна.
    """
    user = f"{complaint}\n\n{scan_context}".strip()
    raw = ollama_client.chat_once(
        [
            {"role": "system", "content": SYSTEM_ROUTER},
            {"role": "user", "content": user},
        ],
        # Маршрутизация — это выбор из списка, а не творчество.
        temperature=0.1,
    )
    chosen = _parse_specialties(raw)
    log.info("Консилиум: %s", ", ".join(s.name for s in chosen))
    return chosen


def opinion_stream(
    specialty: Specialty,
    complaint: str,
    scan_context: str = "",
) -> Iterator[str]:
    """Мнение одного специалиста, по кускам.

    Просим коротко: на консилиуме важна суть, а не развёрнутый приём.
    Полный разговор человек может продолжить во вкладке «Приём».
    """
    system = prompts.build_specialist_prompt(specialty.key)
    system += (
        "\n\n## Формат для консилиума\n"
        "Сейчас ты не ведёшь полный приём, а даёшь короткое мнение коллегам.\n"
        "Уложись в 2–4 предложения: что ты видишь со своей стороны, "
        "насколько это по твоей части и что предлагаешь. Без приветствий."
    )
    user = f"{complaint}\n\n{scan_context}".strip()

    yield from ollama_client.chat_stream(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )


def summarize(complaint: str, opinions: list[Opinion]) -> str:
    """Сводит мнения в общее заключение консилиума.

    Raises:
        ollama_client.OllamaError: модель недоступна.
    """
    body = "\n\n".join(f"{o.specialty.name}: {o.text}" for o in opinions)
    user = f"Жалоба пациента:\n{complaint}\n\nМнения специалистов:\n{body}"

    return ollama_client.chat_once(
        [
            {"role": "system", "content": SYSTEM_SUMMARY},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )


def parse_summary(raw: str) -> dict[str, str]:
    """Разбирает заключение по строкам «КЛЮЧ: значение»."""
    fields = {
        "СОГЛАСИЕ": "",
        "РАЗНОГЛАСИЯ": "",
        "СРОЧНОСТЬ": "",
        "ГЛАВНЫЙ ВРАЧ": "",
        "ПЛАН": "",
    }
    for line in raw.splitlines():
        stripped = line.strip().lstrip("*# ").strip()
        for key in fields:
            if stripped.upper().startswith(f"{key}:"):
                fields[key] = stripped.split(":", 1)[1].strip()
    return fields
