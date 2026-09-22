"""Генерирует JavaScript-движок из правил на Python.

Правила живут в одном месте — ``core/scam_engine.py``. Вторая копия,
правленная руками, разошлась бы с первой на следующей же итерации:
за один день правила менялись пять раз — по реальным записям,
по враждебным перефразировкам и по новым схемам.

Запуск:
    python tools/build_engine_js.py > web/engine.js
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import scam_engine as se  # noqa: E402


def _patterns(items) -> list[dict]:
    return [
        {
            "code": p.code,
            "name": p.name,
            "stems": [list(v) for v in p.stems],
            "weight": p.weight,
            "explanation": p.explanation,
            "decisive": p.decisive,
            "strong": getattr(p, "strong", False),
        }
        for p in items
    ]


RULES = {
    "decisive": _patterns(se.DECISIVE),
    "patterns": _patterns(se.PATTERNS),
    "safe": [
        {"stems": [list(v) for v in variants], "description": d, "relief": r}
        for variants, d, r in se.SAFE_SIGNALS
    ],
    "aliases": {k: list(v) for k, v in se._STEM_ALIASES.items()},
    "window": se.PROXIMITY_WINDOW,
    "thresholds": {
        "watch": se.THRESHOLD_WATCH,
        "alert": se.THRESHOLD_ALERT,
        "danger": se.THRESHOLD_DANGER,
    },
}

TEMPLATE = '''// Движок распознавания схем телефонного мошенничества.
//
// СГЕНЕРИРОВАН из core/scam_engine.py — руками не править.
// Пересобрать: python tools/build_engine_js.py > web/engine.js
//
// Работает целиком на устройстве: ни одного сетевого запроса.
// Схем: %(n_patterns)d, решающих признаков: %(n_decisive)d,
// защитных признаков: %(n_safe)d.

"use strict";

const RULES = %(rules)s;

function normalize(text) {
  return (text || "").toLowerCase().replace(/ё/g, "е")
    .replace(/[^а-яa-zәғқңөұүһі0-9\\s]/g, " ");
}

/** Корень с учётом чередования гласных: болит / побаливает. */
function stemPresent(stem, word) {
  const variants = RULES.aliases[stem] || [stem];
  for (const v of variants) { if (word.indexOf(v) !== -1) return true; }
  return false;
}

/**
 * Корни варианта должны стоять РЯДОМ — в окне из нескольких слов.
 *
 * Без окна они находятся в разных концах текста, и в длинном разговоре
 * пересекаются сами собой: «служба БЕЗОПАСНости банка… по вашему СЧЁТу»
 * срабатывало как «переведите на безопасный счёт».
 */
function variantInWindow(words, stems) {
  if (stems.length === 1) return words.some(w => stemPresent(stems[0], w));
  for (let i = 0; i < words.length; i++) {
    const win = words.slice(i, i + RULES.window);
    if (stems.every(s => win.some(w => stemPresent(s, w)))) return true;
  }
  return false;
}

function matches(words, variants) {
  return variants.some(v => variantInWindow(words, v));
}

const LEVELS = {
  calm:   { weight: 0, label: "Обычный разговор",     emoji: "🟢", warn: false },
  watch:  { weight: 1, label: "Есть подозрительное",  emoji: "🟡", warn: false },
  alert:  { weight: 2, label: "Похоже на мошенников", emoji: "🟠", warn: true  },
  danger: { weight: 3, label: "Это мошенники",        emoji: "🔴", warn: true  },
};

function evaluate(transcript) {
  const words = normalize(transcript).split(/\\s+/).filter(Boolean);
  const matched = [];
  let score = 0;

  for (const p of RULES.decisive) {
    if (matches(words, p.stems)) { matched.push(p); score += p.weight; }
  }
  for (const p of RULES.patterns) {
    if (matches(words, p.stems)) { matched.push(p); score += p.weight; }
  }

  // Сочетание схем опаснее их суммы: обычный разговор редко содержит
  // сразу три признака, а схема обмана — почти всегда.
  if (matched.length >= 3) score += 3;

  const safe = [];
  for (const s of RULES.safe) {
    if (matches(words, s.stems)) { safe.push(s.description); score -= s.relief; }
  }
  score = Math.max(score, 0);

  // Сильный защитный признак гасит решающий: собеседник, который сам
  // отговаривает называть код, мошенником не бывает.
  const hasDecisive = matched.some(p => p.decisive) && safe.length === 0;
  // Самодостаточная схема поднимает тревогу, не дожидаясь суммы баллов:
  // «сколько денег у вас есть» набирало 6 при пороге 8 — система
  // схему распознавала и молчала.
  const hasStrong = matched.some(p => p.strong) && safe.length === 0;

  let level = "calm";
  if (hasDecisive || score >= RULES.thresholds.danger) level = "danger";
  else if (hasStrong || score >= RULES.thresholds.alert) level = "alert";
  else if (score >= RULES.thresholds.watch) level = "watch";

  return { level, score, matched, safe };
}

/** Короткая фраза вслух: в стрессе длинное не воспринимается. */
function warningText(decision) {
  if (!LEVELS[decision.level].warn) return "";
  const top = decision.matched[0];
  if (top && top.decisive) {
    return "Внимание. " + top.name +
      ". Это мошенники. Положите трубку и позвоните в банк сами.";
  }
  return "Внимание. Разговор похож на мошеннический. " +
    "Положите трубку и перезвоните близким.";
}

// Экспорт для Node — чтобы правила можно было прогнать тестами,
// а не только глазами в браузере. В браузере ветка не срабатывает.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { evaluate, warningText, normalize, LEVELS, RULES };
}
'''

print(TEMPLATE % {
    "rules": json.dumps(RULES, ensure_ascii=False, indent=1),
    "n_patterns": len(RULES["patterns"]),
    "n_decisive": len(RULES["decisive"]),
    "n_safe": len(RULES["safe"]),
})
