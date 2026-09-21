"""Уведомление близких.

Самая недооценённая часть продукта. Приложение может сколько угодно
правильно распознать мошенника — но если пожилой человек под давлением
всё равно пойдёт к банкомату, спасти его может только звонок дочери.
Поэтому уведомление детям здесь не дополнение, а вторая половина защиты.

**Канал — SMS, и это осознанный выбор.** Push требует интернета, а мы
работаем там, где его нет. SMS идёт по сотовой сети: она есть везде,
где вообще ловит телефон, и приходит на любой аппарат без приложений.

Отправку выполняет штатное приложение телефона по ссылке ``sms:``.
Мы не подключаем платный шлюз: он потребовал бы интернета и денег,
а главное — увёл бы данные на чужой сервер, чего мы не делаем принципиально.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote

import config
from core.history import Deviation
from core.scam_engine import ScamDecision

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alert:
    """Готовое уведомление."""

    title: str
    body: str
    """Текст сообщения. Уже урезан до длины, разумной для SMS."""

    urgent: bool

    def sms_link(self, phone: str = "") -> str:
        """Ссылка, открывающая приложение сообщений с готовым текстом.

        Формат ``sms:номер?body=текст`` понимают и Android, и iOS.
        Номер необязателен: без него откроется выбор получателя.
        """
        number = phone.strip().replace(" ", "")
        return f"sms:{number}?body={quote(self.body)}"

    def whatsapp_link(self, phone: str = "") -> str:
        """Запасной канал, когда интернет всё же есть."""
        number = "".join(ch for ch in phone if ch.isdigit())
        return f"https://wa.me/{number}?text={quote(self.body)}"


def _timestamp() -> str:
    return datetime.now().strftime("%d.%m в %H:%M")


def scam_alert(decision: ScamDecision, parent_name: str = "") -> Alert:
    """Сообщение о подозрительном звонке.

    Текст построен так, чтобы ребёнок понял ситуацию с первой строки
    и **сразу знал, что делать**. Без конкретного действия уведомление
    только пугает: человек читает «что-то случилось» и не понимает,
    бежать ему или нет.
    """
    who = parent_name.strip() or "Вашему близкому"
    schemes = "; ".join(p.name.lower() for p in decision.matched[:3])

    body = (
        f"⚠️ {config.APP_NAME}\n"
        f"{who} поступил подозрительный звонок {_timestamp()}.\n"
        f"Распознано: {schemes}.\n"
        f"Позвоните и проверьте, всё ли в порядке. "
        f"Попросите не выполнять никаких просьб звонившего."
    )

    return Alert(
        title="Подозрительный звонок",
        body=body,
        urgent=decision.level.weight >= 3,
    )


def health_alert(
    deviation: Deviation,
    bpm: float,
    parent_name: str = "",
) -> Alert:
    """Сообщение о заметном отклонении от личной нормы.

    Отправляется не на любое отклонение, а только на резкое — иначе
    дети начнут получать сообщения через день и перестанут их читать.
    Решение об этом принимает ``history.check_deviation``.
    """
    who = parent_name.strip() or "У вашего близкого"

    body = (
        f"📊 {config.APP_NAME}\n"
        f"{who} замер {_timestamp()}: пульс {bpm:.0f} уд/мин — "
        f"это {deviation.direction} обычной нормы.\n"
        f"Замер сделан камерой и носит оценочный характер. "
        f"Стоит позвонить и спросить о самочувствии."
    )

    return Alert(title="Отклонение от нормы", body=body, urgent=False)


def should_send(decision: ScamDecision | None = None,
                deviation: Deviation | None = None) -> bool:
    """Решает, есть ли повод беспокоить близких.

    Порог намеренно высокий. Уведомление, которое приходит часто,
    перестают читать — и тогда оно не сработает в тот единственный раз,
    когда действительно нужно.
    """
    if decision is not None and decision.level.should_notify:
        return True
    if deviation is not None and deviation.should_notify:
        return True
    return False
