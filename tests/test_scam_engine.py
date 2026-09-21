"""Тесты движка распознавания телефонного мошенничества.

Проверяются две вещи, и вторая важнее первой:

1. Схемы мошенников распознаются в разных формулировках.
2. **Обычные разговоры НЕ вызывают тревогу.** Ложное срабатывание
   на настоящем звонке из банка стоит дороже пропуска: напуганный
   человек выключит приложение и больше не включит.
"""

from __future__ import annotations

import pytest

from core.scam_engine import RiskLevel, evaluate, warning_text


# ─────────────────────────────────────────────────────────────
# Схемы распознаются
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "transcript",
    [
        "Служба безопасности банка. Назовите код из смс, который вам придёт.",
        "Продиктуйте код, который пришёл в сообщении, иначе деньги спишут.",
        "Скажите код подтверждения, я жду на линии.",
        "Банк қызметкері. Кодты айтыңыз, дәл қазір.",
    ],
)
def test_sms_code_request_is_danger(transcript: str) -> None:
    """Просьба назвать код из СМС — решающий признак сама по себе."""
    decision = evaluate(transcript)
    assert decision.level is RiskLevel.DANGER
    assert any(p.code == "sms_code" for p in decision.matched)


@pytest.mark.parametrize(
    "transcript",
    [
        "Переведите деньги на безопасный счёт прямо сейчас.",
        "Нужно перевести средства на резервный счёт банка.",
        "Ақшаны қауіпсіз есепшотқа аударыңыз.",
    ],
)
def test_safe_account_is_danger(transcript: str) -> None:
    """«Безопасного счёта» не существует — верный признак обмана."""
    assert evaluate(transcript).level is RiskLevel.DANGER


def test_relative_in_trouble_scheme() -> None:
    """Схема «родственник попал в беду» с давлением и просьбой молчать."""
    decision = evaluate(
        "Ваш сын попал в аварию, сбил человека. Он сейчас в полиции. "
        "Нужно срочно уладить вопрос, никому не говорите."
    )
    assert decision.level is RiskLevel.DANGER
    codes = {p.code for p in decision.matched}
    assert "relative_trouble" in codes
    assert "secrecy" in codes


def test_remote_access_app_is_danger() -> None:
    """Просьба установить программу удалённого доступа."""
    decision = evaluate("Установите приложение AnyDesk, чтобы мы защитили счёт.")
    assert decision.level is RiskLevel.DANGER
    assert any(p.code == "remote_app" for p in decision.matched)


def test_cvv_request_is_danger() -> None:
    """Просьба назвать три цифры с оборота карты."""
    assert evaluate("Назовите три цифры на обороте карты.").level is RiskLevel.DANGER


def test_several_patterns_raise_score() -> None:
    """Сочетание схем опаснее любой из них по отдельности."""
    single = evaluate("Вас беспокоит служба безопасности банка.")
    combo = evaluate(
        "Служба безопасности банка. Подозрительная операция по счёту. "
        "Срочно, не кладите трубку, никому не говорите."
    )
    assert combo.score > single.score
    assert combo.level.weight > single.level.weight


# ─────────────────────────────────────────────────────────────
# Обычные разговоры не вызывают тревогу — это важнее
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "transcript",
    [
        "Ваша карта готова к выдаче. Подойдите в отделение в удобное время.",
        "Подтвердите запись на приём к терапевту на завтра в десять.",
        "Привет мам, я завтра приеду, купить что-нибудь?",
        "Это банк, изменение тарифа. Перезвоните нам по номеру на обороте карты.",
        "Напоминаем: никому не сообщайте код из смс, даже сотрудникам банка.",
        "Здравствуйте, доставка. Ваш заказ привезут завтра с двух до шести.",
        "Мам, я тут задержусь на работе, приеду поздно.",
    ],
)
def test_normal_conversation_stays_calm(transcript: str) -> None:
    """Ни один обычный разговор не должен поднимать тревогу."""
    decision = evaluate(transcript)
    assert decision.level is RiskLevel.CALM, (
        f"ложное срабатывание ({decision.score:.1f}): {transcript}"
    )
    assert not decision.level.should_warn


def test_bank_advising_callback_is_not_scam() -> None:
    """Совет перезвонить по номеру с карты — самый безопасный из возможных.

    Раньше фраза «номер на обороте карты» срабатывала как требование
    данных карты. Это худший вид ложной тревоги: система пугала человека
    ровно тогда, когда ему давали правильный совет.
    """
    decision = evaluate(
        "Это банк. Перезвоните нам сами по номеру на обороте вашей карты."
    )
    assert decision.level is RiskLevel.CALM
    assert decision.safe_signals


def test_bank_warning_about_code_is_not_scam() -> None:
    """Банк сам предупреждает про код — это защита, а не просьба."""
    decision = evaluate(
        "Банк напоминает: никому не сообщайте код из смс, мы его не спрашиваем."
    )
    assert decision.level is RiskLevel.CALM


def test_safe_signal_suppresses_decisive_pattern() -> None:
    """Сильный защитный признак гасит решающий.

    Промолчать на настоящем звонке дешевле, чем напугать человека.
    """
    decision = evaluate(
        "Никому не сообщайте код из смс. Если сомневаетесь — "
        "перезвоните нам по номеру на обороте карты."
    )
    assert decision.level is RiskLevel.CALM


# ─────────────────────────────────────────────────────────────
# Свойства движка
# ─────────────────────────────────────────────────────────────


def test_decision_is_deterministic() -> None:
    """Одна расшифровка всегда даёт один результат.

    Это и есть главное отличие от языковой модели, и это проверяемо.
    """
    text = "Служба безопасности банка, назовите код из смс."
    results = [evaluate(text) for _ in range(5)]
    assert len({r.level for r in results}) == 1
    assert len({round(r.score, 3) for r in results}) == 1


def test_decision_always_explains_itself() -> None:
    """У каждой тревоги есть объяснение, какие схемы сработали."""
    decision = evaluate("Переведите деньги на безопасный счёт срочно.")
    assert decision.reasons
    assert all(p.explanation for p in decision.matched)


def test_warning_text_names_the_scheme() -> None:
    """Голосовое предупреждение называет конкретную схему, а не просто «тревога»."""
    warning = warning_text(evaluate("Назовите код из смс прямо сейчас."))
    assert "код" in warning.lower()
    assert "трубк" in warning.lower()


def test_no_warning_when_calm() -> None:
    """При спокойном разговоре система молчит."""
    assert warning_text(evaluate("Привет, как дела?")) == ""


@pytest.mark.parametrize("transcript", ["", "   ", "...", "ааа ббб"])
def test_garbage_input_does_not_crash(transcript: str) -> None:
    """Расшифровка приходит кривой — движок не должен падать."""
    decision = evaluate(transcript)
    assert decision.level is RiskLevel.CALM


def test_broken_transcription_still_catches_scheme() -> None:
    """Даже кривая расшифровка ловит схему.

    Распознавание речи ошибается, особенно на казахском. Движок работает
    по корням, поэтому опорные слова схемы проходят и через помехи.
    """
    broken = "служба безопасност банк назовите код смс срочна не кладит трубк"
    assert evaluate(broken).level is RiskLevel.DANGER


def test_language_mix_is_handled() -> None:
    """Мошенники мешают русский и казахский прямо внутри разговора."""
    decision = evaluate("Банк қызметкері, срочно кодты айтыңыз, ақша аудару керек")
    assert decision.level is RiskLevel.DANGER
