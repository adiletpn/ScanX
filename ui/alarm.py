"""Звуковое и голосовое предупреждение на телефоне.

Главная полезная нагрузка продукта. Пожилой человек в момент разговора
смотрит не на экран, а в стену — визуальная плашка его не остановит.
Остановит голос, который прозвучит из телефона поверх разговора.

Как это работает без интернета и без скачивания файлов:

* **Сирена** синтезируется на лету через Web Audio API — два чередующихся
  тона. Никакого звукового файла не нужно.
* **Голос** — встроенный синтезатор речи браузера (``speechSynthesis``).
  На Android и iOS голоса лежат в системе, интернет им не нужен.
  Русский есть на обоих.
* **Вибрация** — на Android. В Safari на iOS её нет, поэтому она
  дополнение, а не основной канал.

Ограничение, которое надо знать: браузеры не дают проигрывать звук,
пока пользователь не коснулся страницы. Нам это не мешает — родитель
нажимает «Включить защиту» в начале, и это засчитывается как касание.
"""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

import config
from core.scam_engine import ScamDecision


def _alarm_html(text: str, urgent: bool) -> str:
    """Собирает страницу, которая звучит и говорит.

    Args:
        text: что произнести вслух.
        urgent: True — резкая сирена, False — мягкий сигнал.
    """
    payload = json.dumps(text, ensure_ascii=False)
    beeps = 3 if urgent else 1
    high, low = (880, 660) if urgent else (620, 520)

    return f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;background:transparent">
<script>
(function () {{
  // ── Сирена: два чередующихся тона через Web Audio API ──
  function siren() {{
    try {{
      var Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      var ctx = new Ctx();
      var t = ctx.currentTime;
      for (var i = 0; i < {beeps}; i++) {{
        [[{high}, 0.0], [{low}, 0.18]].forEach(function (pair) {{
          var osc = ctx.createOscillator();
          var gain = ctx.createGain();
          osc.type = 'square';
          osc.frequency.value = pair[0];
          // Плавный вход и выход: резкий обрыв щёлкает в динамике.
          var start = t + i * 0.42 + pair[1];
          gain.gain.setValueAtTime(0.0001, start);
          gain.gain.exponentialRampToValueAtTime(0.35, start + 0.02);
          gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.16);
          osc.connect(gain); gain.connect(ctx.destination);
          osc.start(start); osc.stop(start + 0.18);
        }});
      }}
    }} catch (e) {{ /* звук недоступен — остаётся голос и экран */ }}
  }}

  // ── Голос: встроенный синтезатор, работает офлайн ──
  function speak() {{
    try {{
      if (!('speechSynthesis' in window)) return;
      var u = new SpeechSynthesisUtterance({payload});
      u.lang = 'ru-RU';
      u.rate = 0.92;    // чуть медленнее обычного: человек в стрессе
      u.pitch = 1.0;
      u.volume = 1.0;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
    }} catch (e) {{ /* голос недоступен — остаётся сирена и экран */ }}
  }}

  // ── Вибрация: только Android, в iOS Safari её нет ──
  function buzz() {{
    try {{
      if (navigator.vibrate) navigator.vibrate([300, 120, 300, 120, 500]);
    }} catch (e) {{}}
  }}

  siren();
  buzz();
  // Голос после сирены: иначе они накладываются и слова тонут.
  setTimeout(speak, {beeps * 420 + 200});
}})();
</script>
</body></html>
"""


def sound_alarm(decision: ScamDecision, spoken: str) -> None:
    """Проигрывает сирену и произносит предупреждение.

    Вызывается один раз на срабатывание. Вставляется скрытым блоком:
    видимую часть рисует ``danger_banner``.
    """
    if not spoken:
        return
    urgent = decision.level.weight >= 3
    components.html(_alarm_html(spoken, urgent), height=0, width=0)


def danger_banner(decision: ScamDecision, spoken: str) -> None:
    """Крупная плашка тревоги во всю ширину экрана.

    Показывает не только «опасно», но и ЧТО именно распознано:
    человек должен понимать, на каком основании его останавливают,
    иначе он решит, что приложение сломалось, и продолжит разговор.
    """
    palette = {
        "danger": (config.COLOR_RED, "🔴", "ЭТО МОШЕННИКИ"),
        "alert": (config.COLOR_AMBER, "🟠", "ПОХОЖЕ НА МОШЕННИКОВ"),
    }
    color, emoji, headline = palette.get(
        decision.level.value, (config.COLOR_AMBER, "🟠", decision.level.label.upper())
    )

    schemes = "".join(
        f"<div style='margin-top:0.35rem'>• {p.name}</div>"
        for p in decision.matched[:4]
    )

    st.markdown(
        f"""
        <div style="border:2px solid {color}; border-radius:18px;
                    background:rgba(255,59,92,0.12); padding:1.1rem 1.2rem;
                    margin:0.6rem 0; box-shadow:0 0 28px rgba(255,59,92,0.3)">
            <div style="text-align:center; font-size:2.6rem; line-height:1">{emoji}</div>
            <div style="text-align:center; font-size:1.5rem; font-weight:800;
                        letter-spacing:0.02em; color:{color}; margin-top:0.3rem">
                {headline}
            </div>
            <div style="text-align:center; font-size:1.05rem; font-weight:700;
                        margin-top:0.6rem">
                Положите трубку
            </div>
            <div style="font-size:0.88rem; line-height:1.5; margin-top:0.8rem;
                        color:{config.COLOR_TEXT}">
                <b>Что распознано:</b>{schemes}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sound_alarm(decision, spoken)


def test_button() -> None:
    """Кнопка проверки звука.

    Нужна не для отладки, а для пользователя: звук в браузере не включится,
    пока страницы не коснулись. Один тест в начале снимает этот вопрос —
    и заодно родитель заранее слышит, как звучит предупреждение.
    """
    if st.button("🔊 Проверить звук предупреждения", key="alarm_test"):
        components.html(
            _alarm_html(
                "Проверка звука. Так прозвучит предупреждение о мошенниках.",
                urgent=False,
            ),
            height=0,
            width=0,
        )
        st.caption("Если вы ничего не услышали — проверьте громкость телефона.")
