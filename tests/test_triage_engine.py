"""Тесты собственного движка триажа.

Движок принимает медицинские решения, поэтому проверяется строже остальных
модулей. Главное требование: **красные флаги не должны пропускаться никогда**,
как бы ни была сформулирована жалоба.
"""

from __future__ import annotations

import pytest

from core.triage_engine import Urgency, evaluate


# ─────────────────────────────────────────────────────────────
# Красные флаги — самое важное
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "complaint",
    [
        "Болит и давит в груди",
        "Давит в груди, отдаёт в руку",
        "Жжение в груди уже час",
        "Онемела левая рука",
        "Речь стала невнятной, нарушение речи",
        "Задыхаюсь, не хватает воздуха",
        "Потерял сознание на работе",
        "Был обморок",
        "Кровотечение сильное, не могу остановить",
        "Отёк горла, тяжело глотать",
    ],
)
def test_red_flags_always_emergency(complaint: str) -> None:
    """Любая формулировка красного флага даёт неотложный статус."""
    decision = evaluate(complaint)
    assert decision.urgency is Urgency.EMERGENCY, f"пропущен флаг: {complaint}"
    assert decision.is_emergency
    assert decision.red_flags, "флаг не записан в решение"


def test_red_flag_beats_everything_else() -> None:
    """Красный флаг перекрывает любые смягчающие обстоятельства."""
    decision = evaluate("Совсем немного побаливает в груди, наверное ерунда")
    assert decision.urgency is Urgency.EMERGENCY


def test_emergency_has_advice() -> None:
    """При неотложке даём указание, что делать до приезда скорой."""
    decision = evaluate("Давит в груди")
    assert all(flag.advice for flag in decision.red_flags)


def test_emergency_is_deterministic() -> None:
    """Одна и та же жалоба всегда даёт один и тот же результат.

    Это главное отличие движка от языковой модели: воспроизводимость.
    """
    results = [evaluate("Онемела рука и перекосило лицо") for _ in range(5)]
    assert all(r.urgency is Urgency.EMERGENCY for r in results)
    assert len({tuple(sorted(f.code for f in r.red_flags)) for r in results}) == 1


# ─────────────────────────────────────────────────────────────
# Маршрутизация
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "complaint,expected",
    [
        ("Голова болит четвёртый день", "neurologist"),
        ("Сыпь на руках, чешется", "dermatologist"),
        ("Зуб болит, десна опухла", "dentist"),
        ("Больно мочиться", "urologist"),
        ("Постоянная жажда", "endocrinologist"),
        ("Изжога после еды", "gastro"),
        ("Тревога и панические атаки", "psychotherapist"),
        ("Нарушился цикл", "gynecologist"),
    ],
)
def test_routing_picks_right_specialist(complaint: str, expected: str) -> None:
    """Жалоба ведёт к профильному специалисту."""
    decision = evaluate(complaint)
    assert decision.specialties, "движок не вернул маршрут"
    assert decision.specialties[0][0] == expected, (
        f"«{complaint}» → {decision.specialties[0][0]}, ожидался {expected}"
    )


def test_vowel_alternation_is_handled() -> None:
    """«побаливает» и «разболелась» — это тоже боль.

    В русском корне чередуется гласная, и без учёта этого половина
    формулировок проходит мимо распознавания.
    """
    for complaint in ("Немного побаливает горло", "Голова разболелась", "Заболело ухо"):
        assert evaluate(complaint).matched, f"не распознано: {complaint}"


def test_unrecognized_complaint_routes_to_therapist() -> None:
    """Непонятную жалобу не замалчиваем, а отправляем к терапевту."""
    decision = evaluate("асдфгх непонятный набор букв")
    assert decision.specialties[0][0] == "therapist"
    assert any("не распознан" in r.lower() for r in decision.reasons)


# ─────────────────────────────────────────────────────────────
# Срочность
# ─────────────────────────────────────────────────────────────


def test_mild_single_symptom_is_self_care() -> None:
    """Одна лёгкая жалоба не требует врача."""
    assert evaluate("Немного побаливает горло со вчера").urgency is Urgency.SELF_CARE


def test_multiple_systems_raise_urgency() -> None:
    """Жалобы по нескольким системам поднимают приоритет."""
    mild = evaluate("Болит горло")
    broad = evaluate("Болит голова, давление скачет, тошнит и отекают ноги")
    assert broad.urgency.weight > mild.urgency.weight


def test_scan_tachycardia_raises_urgency_and_routes_to_cardiology() -> None:
    """Высокий пульс из скана влияет на решение и на маршрут."""
    without = evaluate("Слабость")
    with_scan = evaluate("Слабость", heart_rate_bpm=130.0)

    assert with_scan.score > without.score
    assert any("пульс" in r.lower() for r in with_scan.reasons)
    keys = [k for k, _ in with_scan.specialties]
    assert "cardiologist" in keys


def test_scan_fatigue_is_accounted() -> None:
    """Высокий индекс усталости учитывается в балле."""
    without = evaluate("Слабость")
    with_scan = evaluate("Слабость", fatigue_index=85.0)
    assert with_scan.score > without.score


def test_sudden_onset_raises_score() -> None:
    """Внезапное начало важнее постепенного."""
    gradual = evaluate("Голова болит")
    sudden = evaluate("Голова резко и внезапно заболела")
    assert sudden.score > gradual.score


# ─────────────────────────────────────────────────────────────
# Устойчивость
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("complaint", ["", "   ", "!!!", "123"])
def test_empty_input_does_not_crash(complaint: str) -> None:
    """Пустой или мусорный ввод не должен ронять движок."""
    decision = evaluate(complaint)
    assert decision.urgency in tuple(Urgency)
    assert decision.specialties


def test_case_and_punctuation_are_ignored() -> None:
    """Регистр и знаки препинания не влияют на распознавание."""
    plain = evaluate("болит голова")
    loud = evaluate("БОЛИТ ГОЛОВА!!!  ...")
    assert plain.specialties[0][0] == loud.specialties[0][0]


def test_yo_letter_is_normalized() -> None:
    """«ё» и «е» распознаются одинаково."""
    assert evaluate("отёки на ногах").matched
    assert evaluate("отеки на ногах").matched


def test_decision_always_explains_itself() -> None:
    """У любого решения есть след: без объяснения ему нельзя доверять."""
    for complaint in ("Болит голова", "Давит в груди", "асдфгх"):
        assert evaluate(complaint).reasons, f"нет объяснения для: {complaint}"
