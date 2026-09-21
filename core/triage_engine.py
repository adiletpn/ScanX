"""Собственный движок триажа.

Ключевое архитектурное решение проекта: **решения принимает этот модуль,
а языковая модель только объясняет их человеческим языком.**

Почему не отдать всё модели. Языковая модель на 4 миллиарда параметров
недетерминирована: на один и тот же текст она может дать разный ответ,
и проверить её рассуждение нельзя. В медицинском триаже это неприемлемо —
особенно там, где речь о неотложных состояниях. Пропущенный признак инсульта
не должен зависеть от того, как модели «показалось» в этот раз.

Поэтому:

* **Красные флаги — жёсткие правила.** Сработали — экстренный статус,
  без участия модели. Это гарантия, а не вероятность.
* **Маршрутизация — взвешенное сопоставление.** Каждый симптом даёт вес
  нескольким специальностям, веса складываются, побеждают набравшие больше.
* **Срочность — явная формула** из тяжести симптомов, длительности
  и объективных данных скана.
* **Каждое решение объяснимо.** Движок возвращает список сработавших правил,
  и по нему видно, почему получился именно такой вывод.

Модель подключается последней и только переводит готовое решение
на человеческий язык. Если она недоступна — движок всё равно работает.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

import config

log = logging.getLogger(__name__)


class Urgency(str, Enum):
    """Уровень срочности. Порядок важен: сравниваются по весу."""

    EMERGENCY = "emergency"
    URGENT = "urgent"
    PLANNED = "planned"
    SELF_CARE = "self_care"

    @property
    def weight(self) -> int:
        return {"emergency": 3, "urgent": 2, "planned": 1, "self_care": 0}[self.value]

    @property
    def label(self) -> str:
        return {
            "emergency": "Неотложно",
            "urgent": "Сегодня",
            "planned": "В ближайшие дни",
            "self_care": "Можно справиться самому",
        }[self.value]

    @property
    def emoji(self) -> str:
        return {"emergency": "🚨", "urgent": "🔴", "planned": "🟡", "self_care": "🟢"}[
            self.value
        ]


@dataclass(frozen=True)
class Symptom:
    """Один распознаваемый симптом."""

    code: str
    name: str

    stems: tuple[tuple[str, ...], ...]
    """Варианты распознавания. Каждый вариант — набор корней, которые должны
    встретиться в тексте вместе. Русский язык сильно изменяемый, поэтому
    сравниваем по корням: «болит голова», «головная боль», «голова болела»
    сводятся к паре («голов», «бол»)."""

    specialties: dict[str, float]
    """Вклад в специальности: ключ из реестра → вес."""

    severity: int
    """Базовая тяжесть 0–3. Влияет на итоговую срочность."""


@dataclass(frozen=True)
class RedFlag:
    """Правило неотложного состояния.

    Срабатывание означает экстренный статус без вариантов: языковая модель
    к этому решению не привлекается вообще.
    """

    code: str
    description: str
    stems: tuple[tuple[str, ...], ...]
    advice: str


@dataclass
class TriageDecision:
    """Результат работы движка — полностью объяснимый."""

    urgency: Urgency
    specialties: list[tuple[str, float]]
    """Специальности с набранными весами, по убыванию."""

    matched: list[Symptom] = field(default_factory=list)
    red_flags: list[RedFlag] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    """Человекочитаемый след решения — что именно на него повлияло."""

    score: float = 0.0
    """Итоговый балл срочности."""

    @property
    def is_emergency(self) -> bool:
        return self.urgency is Urgency.EMERGENCY


# ─────────────────────────────────────────────────────────────
# Красные флаги
# ─────────────────────────────────────────────────────────────

RED_FLAGS: tuple[RedFlag, ...] = (
    RedFlag(
        code="chest_pain",
        description="боль или давление в груди",
        stems=(("груд", "бол"), ("груд", "дав"), ("груд", "жж"), ("сердц", "бол")),
        advice="Сядьте, не напрягайтесь, откройте окно. Не садитесь за руль.",
    ),
    RedFlag(
        code="stroke",
        description="признаки инсульта: слабость или онемение одной стороны, "
                    "перекошенное лицо, нарушение речи",
        stems=(
            ("онемел",), ("онемен",),
            ("перекос",), ("речь", "наруш"), ("говор", "не мог"),
            ("слабост", "сторон"), ("рука", "отня"), ("ног", "отня"),
        ),
        advice="Запомните время начала симптомов — это важно для врачей. "
               "Не давайте есть и пить.",
    ),
    RedFlag(
        code="breathing",
        description="сильное затруднение дыхания",
        stems=(("задых",), ("удуш",), ("дыш", "не мог"), ("воздух", "не хват")),
        advice="Сядьте прямо, расстегните одежду, откройте окно.",
    ),
    RedFlag(
        code="consciousness",
        description="потеря или спутанность сознания",
        stems=(("сознан", "потер"), ("обморок",), ("отключ", "созна"), ("без созна",)),
        advice="Уложите на бок. Не оставляйте одного.",
    ),
    RedFlag(
        code="bleeding",
        description="сильное кровотечение",
        stems=(("кровотеч", "сильн"), ("кров", "остановит"), ("кров", "не останав")),
        advice="Прижмите рану, поднимите повреждённое место выше уровня сердца.",
    ),
    RedFlag(
        code="self_harm",
        description="мысли о причинении вреда себе",
        stems=(("не хоч", "жить"), ("покончит",), ("суицид",), ("себе", "вред")),
        advice="Вы не одни. Позвоните близкому человеку прямо сейчас "
               "и не оставайтесь в одиночестве.",
    ),
    RedFlag(
        code="anaphylaxis",
        description="отёк лица, губ или горла",
        stems=(("отёк", "горл"), ("отек", "горл"), ("отёк", "губ"), ("отек", "лиц")),
        advice="Если есть назначенный врачом адреналин — примените по инструкции.",
    ),
)


# ─────────────────────────────────────────────────────────────
# Симптомы и маршрутизация
# ─────────────────────────────────────────────────────────────

SYMPTOMS: tuple[Symptom, ...] = (
    Symptom("headache", "головная боль",
            (("голов", "бол"), ("мигрен",)),
            {"neurologist": 3.0, "therapist": 1.0, "ophthalmologist": 0.5}, 1),
    Symptom("dizziness", "головокружение",
            (("головокруж",), ("кружит", "голов")),
            {"neurologist": 2.5, "cardiologist": 1.5, "ent": 1.0}, 2),
    Symptom("pressure", "проблемы с давлением",
            (("давлен",), ("гипертон",)),
            {"cardiologist": 3.0, "therapist": 1.0, "nephrologist": 1.0}, 2),
    Symptom("palpitations", "перебои в сердце",
            (("сердц", "колот"), ("пульс", "част"), ("аритм",), ("сердцебиен",)),
            {"cardiologist": 3.5, "endocrinologist": 0.5}, 2),
    Symptom("fever", "повышенная температура",
            (("температур",), ("жар",), ("озноб",)),
            {"therapist": 2.5, "infectionist": 1.5, "pediatrician": 0.5}, 1),
    Symptom("cough", "кашель",
            (("кашл",), ("кашел",)),
            {"pulmonologist": 2.5, "therapist": 1.5, "ent": 0.5}, 1),
    Symptom("breath_short", "одышка",
            (("одышк",), ("дыхан", "тяжел")),
            {"pulmonologist": 3.0, "cardiologist": 2.0}, 2),
    Symptom("abdominal", "боль в животе",
            (("живот", "бол"), ("желуд", "бол")),
            {"gastro": 3.0, "therapist": 1.0, "urologist": 0.5}, 2),
    Symptom("heartburn", "изжога",
            (("изжог",), ("кислот", "рот")),
            {"gastro": 3.0}, 1),
    Symptom("nausea", "тошнота",
            (("тошн",), ("рвот",)),
            {"gastro": 2.0, "neurologist": 1.0, "infectionist": 1.0}, 1),
    Symptom("rash", "сыпь или изменения кожи",
            (("сыпь",), ("зуд",), ("пятн", "кож"), ("родин",)),
            {"dermatologist": 3.5, "allergist": 1.5}, 1),
    Symptom("joint_pain", "боль в суставах",
            (("сустав",), ("колен", "бол"), ("плеч", "бол")),
            {"orthopedist": 3.0, "rheumatologist": 2.0}, 1),
    Symptom("back_pain", "боль в спине",
            (("спин", "бол"), ("поясниц", "бол")),
            {"orthopedist": 2.5, "neurologist": 1.5, "nephrologist": 1.0}, 1),
    Symptom("throat", "боль в горле",
            (("горл", "бол"), ("глотат", "больно")),
            {"ent": 3.0, "therapist": 1.0}, 1),
    Symptom("vision", "проблемы со зрением",
            (("зрен",), ("глаз", "бол"), ("вид", "плохо")),
            {"ophthalmologist": 3.5, "neurologist": 1.0}, 2),
    Symptom("urination", "проблемы с мочеиспусканием",
            (("мочеиспуск",), ("мочит", "больно"), ("моч", "кров")),
            {"urologist": 3.5, "nephrologist": 1.5}, 2),
    Symptom("anxiety", "тревога или подавленность",
            (("тревог",), ("паническ",), ("депресс",), ("настроен", "плох")),
            {"psychotherapist": 3.5, "therapist": 0.5}, 1),
    Symptom("insomnia", "нарушения сна",
            (("бессонниц",), ("сон", "плох"), ("не мог", "усну"), ("не сплю",)),
            {"psychotherapist": 2.0, "neurologist": 1.5, "therapist": 0.5}, 1),
    Symptom("weight", "изменение веса",
            (("вес", "терял"), ("вес", "набра"), ("похуде",), ("поправил",)),
            {"endocrinologist": 3.0, "oncologist": 1.0, "gastro": 0.5}, 2),
    Symptom("thirst", "сильная жажда",
            (("жажд",), ("пить", "хоч", "постоянн")),
            {"endocrinologist": 3.5, "nephrologist": 1.0}, 2),
    Symptom("fatigue", "слабость и утомляемость",
            (("слабост",), ("устал",), ("сил", "нет"), ("утомля",)),
            {"therapist": 2.0, "endocrinologist": 1.5, "psychotherapist": 1.0}, 1),
    Symptom("swelling", "отёки",
            (("отёк",), ("отек",), ("опух",)),
            {"nephrologist": 2.5, "cardiologist": 2.0, "allergist": 1.0}, 2),
    Symptom("toothache", "зубная боль",
            (("зуб", "бол"), ("десн",)),
            {"dentist": 4.0}, 1),
    Symptom("ear", "боль в ухе или снижение слуха",
            (("ух", "бол"), ("слух",), ("ушн",)),
            {"ent": 3.5}, 1),
    Symptom("cycle", "нарушения цикла",
            (("менструац",), ("цикл", "наруш"), ("месячн",)),
            {"gynecologist": 4.0}, 1),
    Symptom("allergy", "аллергическая реакция",
            (("аллерг",), ("чиха",), ("нос", "залож")),
            {"allergist": 3.0, "ent": 1.5}, 1),
    Symptom("lump", "уплотнение или образование",
            (("уплотнен",), ("шишк",), ("образован",)),
            {"oncologist": 3.0, "therapist": 1.0}, 2),
)


# ─────────────────────────────────────────────────────────────
# Распознавание текста
# ─────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Приводит текст к виду, удобному для поиска корней.

    Убираем регистр, ё→е и всю пунктуацию: пользователь пишет как придётся,
    и «Голова БОЛИТ!!!» должно распознаваться так же, как «голова болит».
    """
    lowered = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return re.sub(r"[^а-яa-z0-9\s]", " ", lowered)


#: Чередования гласных в корне: «болит» / «побаливает», «тёк» / «отекает».
#: Без этого «немного побаливает горло» не распознаётся как боль в горле.
_STEM_ALIASES: dict[str, tuple[str, ...]] = {
    "бол": ("бол", "бал"),
    "отек": ("отек", "отёк"),
    "дав": ("дав", "давл"),
}


def _stem_present(stem: str, text: str) -> bool:
    """Проверяет корень с учётом чередований."""
    for variant in _STEM_ALIASES.get(stem, (stem,)):
        if variant in text:
            return True
    return False


def _matches(text: str, variants: tuple[tuple[str, ...], ...]) -> bool:
    """Проверяет, встретился ли хотя бы один вариант целиком."""
    for variant in variants:
        if all(_stem_present(_normalize(stem).strip(), text) for stem in variant):
            return True
    return False


def _duration_bonus(text: str) -> tuple[float, str | None]:
    """Оценивает вклад длительности жалобы.

    Долго тянущаяся жалоба повышает приоритет: то, что не проходит неделями,
    уже не «само пройдёт». Внезапное начало — тоже, но по другой причине.
    """
    if _matches(text, (("внезапн",), ("резко",), ("вдруг",))):
        return 1.5, "внезапное начало"
    if _matches(text, (("месяц",), ("год",), ("давно",), ("недел",))):
        return 1.0, "длится долго"
    return 0.0, None


# ─────────────────────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────────────────────

def evaluate(
    complaint: str,
    heart_rate_bpm: float | None = None,
    fatigue_index: float | None = None,
) -> TriageDecision:
    """Принимает решение по жалобе и объективным данным скана.

    Args:
        complaint: текст жалобы от пользователя.
        heart_rate_bpm: пульс из скана, если замер прошёл контроль качества.
        fatigue_index: индекс усталости 0–100.

    Returns:
        TriageDecision со срочностью, маршрутом и следом принятия решения.
    """
    text = _normalize(complaint)

    # ── 1. Красные флаги: жёсткие правила, вне очереди ──
    flags = [flag for flag in RED_FLAGS if _matches(text, flag.stems)]
    if flags:
        log.info("Красный флаг: %s", ", ".join(f.code for f in flags))
        return TriageDecision(
            urgency=Urgency.EMERGENCY,
            specialties=[("therapist", 1.0)],
            red_flags=flags,
            reasons=[f"Неотложный признак: {f.description}" for f in flags],
            score=100.0,
        )

    # ── 2. Симптомы и маршрутизация ──
    matched = [s for s in SYMPTOMS if _matches(text, s.stems)]

    weights: dict[str, float] = {}
    for symptom in matched:
        for key, weight in symptom.specialties.items():
            weights[key] = weights.get(key, 0.0) + weight

    reasons: list[str] = []
    if matched:
        reasons.append("Распознано: " + ", ".join(s.name for s in matched))

    # ── 3. Балл срочности ──
    score = float(sum(s.severity for s in matched))

    bonus, note = _duration_bonus(text)
    score += bonus
    if note:
        reasons.append(f"Учтено: {note}")

    # Несколько систем сразу — повод отнестись серьёзнее.
    if len({key for s in matched for key in s.specialties}) >= 4:
        score += 1.0
        reasons.append("Жалобы затрагивают несколько систем")

    # ── 4. Объективные данные скана ──
    if heart_rate_bpm is not None:
        low, high = config.HR_RESTING_NORMAL
        if heart_rate_bpm > high + 20:
            score += 2.0
            weights["cardiologist"] = weights.get("cardiologist", 0.0) + 2.0
            reasons.append(f"Скан: пульс {heart_rate_bpm:.0f} — заметно выше нормы покоя")
        elif heart_rate_bpm > high:
            score += 1.0
            weights["cardiologist"] = weights.get("cardiologist", 0.0) + 1.0
            reasons.append(f"Скан: пульс {heart_rate_bpm:.0f} — выше нормы покоя")
        elif heart_rate_bpm < low - 10:
            score += 1.5
            weights["cardiologist"] = weights.get("cardiologist", 0.0) + 1.5
            reasons.append(f"Скан: пульс {heart_rate_bpm:.0f} — ниже нормы покоя")

    if fatigue_index is not None and fatigue_index > config.FATIGUE_MODERATE_MAX:
        score += 1.0
        reasons.append(f"Скан: высокий индекс усталости ({fatigue_index:.0f} из 100)")

    # ── 5. Срочность по баллу ──
    if score >= 7.0:
        urgency = Urgency.URGENT
    elif score >= 3.0:
        urgency = Urgency.PLANNED
    elif matched:
        urgency = Urgency.SELF_CARE
    else:
        # Ничего не распознали — не делаем вид, что разобрались.
        urgency = Urgency.PLANNED
        reasons.append("Жалоба не распознана — нужен осмотр врача общей практики")
        weights["therapist"] = weights.get("therapist", 0.0) + 1.0

    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)

    log.info(
        "Триаж: %s (балл %.1f), маршрут %s",
        urgency.value, score, [k for k, _ in ranked[:3]],
    )

    return TriageDecision(
        urgency=urgency,
        specialties=ranked,
        matched=matched,
        reasons=reasons,
        score=score,
    )
