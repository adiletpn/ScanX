"""Экран «Защитник» — живое прослушивание разговора.

Устроен вокруг одного ограничения Streamlit: он перезапускает скрипт
на каждое действие, а нам нужен непрерывный звук. Поэтому слушатель
живёт в ``st.cache_resource`` — так он переживает перерисовки, — а сам
экран обновляется фрагментом с автоповтором, который перечитывает
состояние и перерисовывает только себя.

Тревога показывается один раз на срабатывание: без этого при каждом
обновлении экрана заново звучала бы сирена.
"""

from __future__ import annotations

import logging

import streamlit as st

import config
from core import alerts, audio_io, registry
from core.scam_engine import RiskLevel, ScamDecision, evaluate, warning_text
from ui import alarm
from ui import components as ui

log = logging.getLogger(__name__)


@st.cache_resource
def _listener() -> audio_io.MicrophoneListener:
    """Один слушатель на всё приложение.

    ``cache_resource`` — единственный способ пережить перезапуск скрипта:
    обычные переменные и session_state потока не удержат.
    """
    return audio_io.MicrophoneListener()


def _risk_meter(decision: ScamDecision) -> None:
    """Шкала риска с разбором: что просят и как давят.

    Две шкалы, а не одна. Подозрительная просьба и манипуляция — разные
    вещи, и видеть их раздельно полезнее: человек понимает не только
    «опасно», но и чем именно его обрабатывают.
    """
    palette = {
        "calm": config.COLOR_GREEN,
        "watch": config.COLOR_AMBER,
        "alert": config.COLOR_AMBER,
        "danger": config.COLOR_RED,
    }
    color = palette[decision.level.value]

    # Схемы давления — то, КАК обрабатывают, в отличие от того, ЧТО просят.
    pressure_codes = {"urgency", "secrecy", "dont_hang_up", "relative_trouble"}
    pressure = [p for p in decision.matched if p.code in pressure_codes]
    demands = [p for p in decision.matched if p.code not in pressure_codes]

    filled = min(int(decision.score / 12.0 * 100), 100)

    st.markdown(
        f"""
        <div class="sx-card">
            <div style="display:flex; justify-content:space-between;
                        align-items:baseline">
                <span class="sx-metric-label">Оценка разговора</span>
                <span style="font-size:1.05rem; font-weight:700; color:{color}">
                    {decision.level.emoji} {decision.level.label}
                </span>
            </div>
            <div style="height:12px; border-radius:99px; margin-top:0.6rem;
                        background:rgba(255,255,255,0.07); overflow:hidden">
                <div style="width:{filled}%; height:12px; border-radius:99px;
                            background:{color}; box-shadow:0 0 14px {color}"></div>
            </div>
            <div style="display:flex; gap:1.2rem; margin-top:0.8rem;
                        font-size:0.85rem">
                <div style="flex:1">
                    <div style="color:{config.COLOR_TEXT_DIM}">Что просят</div>
                    <div style="font-weight:700; margin-top:0.2rem">
                        {len(demands)} призн.
                    </div>
                </div>
                <div style="flex:1">
                    <div style="color:{config.COLOR_TEXT_DIM}">Как давят</div>
                    <div style="font-weight:700; margin-top:0.2rem;
                                color:{config.COLOR_MAGENTA if pressure else config.COLOR_TEXT}">
                        {len(pressure)} приём.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if decision.matched:
        with st.expander("Что именно распознано", expanded=decision.level.should_warn):
            for pattern in decision.matched:
                st.markdown(f"**{pattern.name}** — {pattern.explanation}")
            for signal in decision.safe_signals:
                st.markdown(f"✓ _{signal}_")


def _notify_block(alert: alerts.Alert) -> None:
    """Готовое сообщение близким — одним касанием.

    Отправку делает штатное приложение сообщений по ссылке ``sms:``.
    Платный шлюз здесь был бы неуместен: он требует интернета и увёл бы
    данные на чужой сервер, а мы работаем офлайн принципиально.
    """
    phone = st.session_state.get("child_phone", "")
    st.markdown(
        f"""
        <div class="sx-card" style="border-color:{config.COLOR_MAGENTA}">
            <div class="sx-metric-label">📩 Сообщить близким</div>
            <div style="font-size:0.85rem; line-height:1.55; margin-top:0.5rem;
                        white-space:pre-line">{alert.body}</div>
            <a href="{alert.sms_link(phone)}"
               style="display:block; text-align:center; margin-top:0.8rem;
                      padding:0.7rem; border-radius:12px; text-decoration:none;
                      font-weight:700; color:#02131a; background:{config.COLOR_MAGENTA}">
                Отправить SMS
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _report_block(decision: ScamDecision) -> None:
    """Отправка номера в реестр — то, чем Android защищает iPhone.

    Разобрать разговор на iOS невозможно: Apple не даёт микрофон во время
    звонка. Но предупредить ДО ответа можно — через Call Directory, тем же
    механизмом, что у определителей номера. Номера для этого списка
    поставляют те, у кого приложение разговор слышит.
    """
    with st.expander("📇 Добавить номер в реестр"):
        st.caption(
            "Номер попадёт в общий список. Владельцы iPhone увидят "
            "предупреждение ещё до того, как снимут трубку — там разобрать "
            "разговор нельзя, но предупредить о номере можно."
        )
        number = st.text_input(
            "Номер звонившего",
            key="scam_number",
            placeholder="+7 700 000 00 00",
            label_visibility="collapsed",
        )
        if st.button("Добавить в реестр", key="registry_add"):
            record = registry.report(number, [p.code for p in decision.matched])
            if record is None:
                st.warning("Не разобрал номер.")
            elif record.is_confirmed:
                st.success(
                    f"Номер подтверждён ({record.reports} сообщения). "
                    "Владельцы iPhone будут предупреждены."
                )
            else:
                left = config.REGISTRY_MIN_REPORTS - record.reports
                st.info(
                    f"Записано. Нужно ещё {left} подтверждение от других "
                    "пользователей — один сигнал номер не осуждает: "
                    "мошенники подменяют чужие номера."
                )


def _level_bar(levels: list[float]) -> None:
    """Индикатор громкости — по нему видно, что микрофон живой."""
    if not levels:
        return
    bars = "".join(
        f'<div style="flex:1; height:{max(int(v * 34), 2)}px; '
        f'background:{config.COLOR_CYAN}; opacity:{0.35 + v * 0.65}; '
        f'border-radius:2px"></div>'
        for v in levels[-30:]
    )
    st.markdown(
        f'<div style="display:flex; align-items:flex-end; gap:2px; height:36px; '
        f'margin:0.3rem 0">{bars}</div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every=config.GUARD_REFRESH_SEC)
def _live_panel() -> None:
    """Живая часть экрана: обновляется сама, не трогая остальную страницу."""
    listener = _listener()
    state = listener.snapshot()

    if state.error:
        ui.note(state.error, kind="danger")
        return

    if not state.running:
        ui.note("Защита выключена. Нажмите «Включить защиту».")
        return

    heard = f"{state.transcript} {state.partial}".strip()
    decision = evaluate(heard)

    # Тревога звучит один раз на повышение уровня: иначе сирена
    # повторялась бы при каждом обновлении экрана.
    previous = st.session_state.get("guard_last_level", RiskLevel.CALM)
    if decision.level.should_warn and decision.level.weight > previous.weight:
        alarm.danger_banner(decision, warning_text(decision))
        st.session_state.guard_last_level = decision.level
    elif decision.level.should_warn:
        # Уровень не вырос — плашку показываем, но сирену не повторяем.
        ui.note(
            f"{decision.level.emoji} <b>{decision.level.label}</b> — положите трубку",
            kind="danger",
        )

    _risk_meter(decision)

    if alerts.should_send(decision=decision):
        _notify_block(alerts.scam_alert(decision, st.session_state.get("parent_name", "")))
        _report_block(decision)

    _level_bar(state.levels)

    st.caption(f"Слушаю {state.seconds:.0f} с · обработано {state.chunks} фрагментов")

    st.markdown("**Что слышно**")
    if heard:
        st.markdown(
            f'<div class="sx-card" style="font-size:0.9rem; line-height:1.6; '
            f'max-height:220px; overflow-y:auto">{heard}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Пока тихо…")


def render() -> None:
    """Рисует вкладку «Защитник»."""
    listener = _listener()
    state = listener.snapshot()

    ui.note(
        "Включите громкую связь и положите телефон рядом. "
        "Приложение слушает разговор и предупредит, если распознает "
        "приёмы мошенников."
    )

    left, right = st.columns(2)
    with left:
        if not state.running:
            if st.button("🛡 Включить защиту", type="primary", key="guard_start"):
                st.session_state.guard_last_level = RiskLevel.CALM
                try:
                    listener.start()
                except audio_io.AudioError as exc:
                    ui.note(str(exc).replace("\n", "<br>"), kind="danger")
                else:
                    st.rerun()
        else:
            if st.button("■ Остановить", key="guard_stop"):
                listener.stop()
                st.rerun()

    with right:
        if st.button("↻ Новый разговор", key="guard_reset"):
            listener.reset()
            st.session_state.guard_last_level = RiskLevel.CALM
            st.rerun()

    alarm.test_button()

    with st.expander("Кому сообщать"):
        st.text_input(
            "Имя родителя",
            key="parent_name",
            placeholder="Маме",
            help="Подставится в сообщение, чтобы ребёнок сразу понял, о ком речь",
        )
        st.text_input(
            "Телефон близкого",
            key="child_phone",
            placeholder="+7 700 000 00 00",
            help="Сообщение уйдёт обычной SMS — она работает и без интернета",
        )

    _live_panel()

    with st.expander("Проверить на записи разговора"):
        st.caption(
            "Запасной вариант, если звук в зале не берётся: "
            "загрузите запись звонка, и она пройдёт тот же разбор."
        )
        uploaded = st.file_uploader(
            "Запись разговора",
            type=["wav", "mp3", "m4a", "aac", "ogg", "aiff"],
            key="guard_file",
            label_visibility="collapsed",
        )
        if uploaded is not None and st.button("Разобрать запись", key="guard_file_run"):
            _analyze_file(uploaded)


def _analyze_file(uploaded) -> None:
    """Разбирает загруженную запись — запасной путь для демонстрации."""
    import tempfile
    from pathlib import Path

    from core.asr import AsrError, transcribe_file

    suffix = Path(uploaded.name).suffix or ".wav"
    tmp = Path(tempfile.mkdtemp(prefix="scanx_")) / f"call{suffix}"
    tmp.write_bytes(uploaded.getvalue())

    with st.spinner("Слушаю запись…"):
        try:
            text = transcribe_file(tmp)
        except AsrError as exc:
            ui.note(str(exc), kind="warn")
            return
        except Exception:  # noqa: BLE001 — граница модуля
            log.exception("Сбой разбора записи")
            ui.note("Не удалось прочитать запись. Попробуйте другой файл.", kind="warn")
            return

    decision = evaluate(text)
    if decision.level.should_warn:
        alarm.danger_banner(decision, warning_text(decision))
    _risk_meter(decision)

    st.markdown("**Расшифровка**")
    st.markdown(
        f'<div class="sx-card" style="font-size:0.9rem; line-height:1.6">'
        f'{text or "(речь не распознана)"}</div>',
        unsafe_allow_html=True,
    )
