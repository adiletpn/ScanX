"""Проверка на настоящих записях звонков.

Эти тесты — единственные, которые работают с живой речью, а не с текстом,
который я написал сам. Всё остальное проверяет логику; эти проверяют,
что логика совпадает с реальностью.

Записи лежат в ``demo/calls/`` и в репозиторий не попадают: это голоса
конкретных людей. Если папки нет, тесты пропускаются — чтобы проект
собирался у любого, кто его склонировал.

Настройка движка делалась ПО ЭТИМ записям, и дважды она расходилась
с тем, что казалось очевидным на бумаге:

* мошенник в схеме «родственник в беде» говорит от ПЕРВОГО лица —
  «мама, это я, я попал в аварию», а не «ваш сын попал в аварию»;
* вопрос «сколько вы можете собрать» не был заведён вовсе, хотя это
  один из сильнейших признаков: ни банк, ни полиция его не задают.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.asr import AsrError, transcribe_file
from core.scam_engine import RiskLevel, evaluate

CALLS_DIR = Path(__file__).parent.parent / "demo" / "calls"

#: Ожидания по каждой записи. Имена дефолтные — так их отдал диктофон.
EXPECTED: dict[str, tuple[str, RiskLevel]] = {
    "New Recording.m4a": ("служба безопасности банка", RiskLevel.DANGER),
    "New Recording 2.m4a": ("родственник в беде", RiskLevel.DANGER),
    "New Recording 3.m4a": ("безопасный счёт", RiskLevel.DANGER),
    "New Recording 4.m4a": ("казахский, код из СМС", RiskLevel.DANGER),
    "New Recording 5.m4a": ("удалённый доступ, смешанный язык", RiskLevel.DANGER),
    "New Recording 6.m4a": ("настоящий банк", RiskLevel.CALM),
    "New Recording 7.m4a": ("сын звонит маме", RiskLevel.CALM),
    "New Recording 8.m4a": ("поликлиника, казахский", RiskLevel.CALM),
}


def _available() -> list[str]:
    if not CALLS_DIR.exists():
        return []
    return [name for name in EXPECTED if (CALLS_DIR / name).exists()]


pytestmark = pytest.mark.skipif(
    not _available(),
    reason="нет записей в demo/calls (они не хранятся в репозитории)",
)


@pytest.mark.parametrize("filename", list(EXPECTED))
def test_real_call_is_classified_correctly(filename: str) -> None:
    """Каждая запись получает ожидаемый уровень тревоги."""
    path = CALLS_DIR / filename
    if not path.exists():
        pytest.skip(f"нет файла {filename}")

    description, expected = EXPECTED[filename]

    try:
        transcript = transcribe_file(path, dual=True)
    except AsrError as exc:
        pytest.skip(f"распознавание недоступно: {exc}")

    decision = evaluate(transcript)

    assert decision.level is expected, (
        f"«{description}»: получили {decision.level.value} "
        f"(балл {decision.score:.1f}) вместо {expected.value}\n"
        f"расшифровка: {transcript[:200]}"
    )


def test_no_false_alarm_on_any_normal_call() -> None:
    """Ни одна нормальная запись не вызывает тревогу.

    Проверяется отдельно от остальных: ложное срабатывание на настоящем
    звонке из банка дороже пропуска. Напуганный человек выключит
    приложение и больше не включит.
    """
    normals = [n for n, (_d, lvl) in EXPECTED.items() if lvl is RiskLevel.CALM]
    checked = 0

    for name in normals:
        path = CALLS_DIR / name
        if not path.exists():
            continue
        try:
            decision = evaluate(transcribe_file(path, dual=True))
        except AsrError:
            pytest.skip("распознавание недоступно")
        assert not decision.level.should_warn, (
            f"ложная тревога на записи {name}: {decision.level.value}"
        )
        checked += 1

    if checked == 0:
        pytest.skip("нормальных записей нет")


def test_all_scam_calls_trigger_a_spoken_warning() -> None:
    """Каждая мошенническая запись доводит дело до голосового предупреждения."""
    scams = [n for n, (_d, lvl) in EXPECTED.items() if lvl is RiskLevel.DANGER]
    checked = 0

    for name in scams:
        path = CALLS_DIR / name
        if not path.exists():
            continue
        try:
            decision = evaluate(transcribe_file(path, dual=True))
        except AsrError:
            pytest.skip("распознавание недоступно")
        assert decision.level.should_warn, f"нет предупреждения на {name}"
        assert decision.matched, f"нет объяснения на {name}"
        checked += 1

    if checked == 0:
        pytest.skip("мошеннических записей нет")
