"""ScanX — точка входа Streamlit.

Медицинский помощник первого шага: скан по видео лица, ИИ-ассистент
и маршрутизация к врачу. Работает полностью офлайн.

Запуск:
    streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

import config
from ai import ollama_client, prompts
from core import scan as scan_mod
from core import video_io
from ui import components as ui
from ui.styles import inject_styles, render_header

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("scanx")


def init_session_state() -> None:
    """Готовит хранилище сессии.

    Streamlit перезапускает скрипт на каждое действие пользователя,
    поэтому всё, что должно пережить перерисовку, живёт в session_state.
    """
    defaults = {
        "scan_result": None,       # последний результат скана
        "baseline_scan": None,     # замер в покое — для пробы с нагрузкой
        "chat_messages": [],       # история диалога с ассистентом
        "triage_result": None,     # карточка маршрутизации
        "jump_to_chat": False,     # переход из скана в чат с контекстом
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────────────────────
# Вкладка «Скан»
# ─────────────────────────────────────────────────────────────


def _process_video(path: Path) -> None:
    """Обрабатывает видео и кладёт результат в session_state."""
    progress_bar = st.progress(0.0, text="Начинаю…")

    def on_progress(fraction: float, label: str) -> None:
        progress_bar.progress(min(fraction, 1.0), text=label)

    try:
        result = scan_mod.run_scan(path, progress=on_progress)
    except scan_mod.ScanError as exc:
        progress_bar.empty()
        ui.note(str(exc), kind="warn")
        return
    except Exception:  # noqa: BLE001 — последний рубеж, падать нельзя
        progress_bar.empty()
        log.exception("Непредвиденный сбой скана")
        ui.note(
            "Что-то пошло не так при обработке. Попробуйте другое видео.",
            kind="danger",
        )
        return

    progress_bar.empty()
    st.session_state.scan_result = result

    # Первый удачный замер становится точкой отсчёта для пробы с нагрузкой.
    if (
        st.session_state.baseline_scan is None
        and result.quality.level.is_trustworthy
    ):
        st.session_state.baseline_scan = result


def _render_scan_result() -> None:
    """Карточка результата последнего замера."""
    result = st.session_state.scan_result
    if result is None:
        return

    st.divider()

    ui.heart_rate_card(result.heart_rate, result.quality)
    ui.fatigue_card(result.fatigue)
    ui.quality_card(result.quality)

    # Проба с нагрузкой: показываем сравнение, только если есть два разных
    # замера и обоим можно доверять.
    baseline = st.session_state.baseline_scan
    if (
        baseline is not None
        and baseline is not result
        and baseline.quality.level.is_trustworthy
        and result.quality.level.is_trustworthy
    ):
        ui.comparison_card(baseline.heart_rate.bpm, result.heart_rate.bpm)

    ui.preview_image(result.preview)

    with st.expander("Графики сигнала"):
        ui.pulse_wave_chart(result.heart_rate)
        ui.spectrum_chart(result.heart_rate)
        st.caption(
            f"Обработано за {result.processing_sec:.0f} с · "
            f"длительность записи {result.duration_sec:.0f} с"
        )

    if st.button("Спросить ИИ о результатах", key="ask_ai"):
        st.session_state.jump_to_chat = True
        st.session_state.chat_messages.append(
            {"role": "user", "content": "Что значат мои результаты?"}
        )
        st.rerun()

    ui.disclaimer()


def render_scan_tab() -> None:
    """Вкладка «Скан» — загрузка видео и результаты замера."""
    ui.note(
        f"<b>Как записать</b><br>"
        f"Лицо в кадре, ровный свет спереди. "
        f"Не двигайтесь и не говорите {config.RECOMMENDED_DURATION_SEC} секунд."
    )

    uploaded = st.file_uploader(
        "Видео лица",
        type=config.ALLOWED_VIDEO_TYPES,
        key="scan_video",
        label_visibility="collapsed",
    )

    if uploaded is not None and st.button("Начать анализ", type="primary"):
        suffix = Path(uploaded.name).suffix or ".mp4"
        try:
            path = video_io.save_upload_to_temp(uploaded.getvalue(), suffix=suffix)
        except video_io.VideoReadError as exc:
            ui.note(str(exc), kind="warn")
        else:
            _process_video(path)

    demo_path = scan_mod.find_demo_video()
    if demo_path is not None:
        if st.button(f"Демо-видео ({demo_path.name})", key="scan_demo"):
            _process_video(demo_path)

    if st.session_state.baseline_scan is not None and st.button(
        "Сбросить замер в покое", key="reset_baseline"
    ):
        st.session_state.baseline_scan = None
        st.rerun()

    _render_scan_result()


# ─────────────────────────────────────────────────────────────
# Вкладка «Ассистент»
# ─────────────────────────────────────────────────────────────


def _build_chat_messages() -> list[dict[str, object]]:
    """Собирает историю с системным промптом и контекстом скана."""
    system = prompts.SYSTEM_ASSISTANT

    result = st.session_state.scan_result
    if result is not None:
        context = prompts.format_scan_context(
            result.heart_rate, result.fatigue, result.quality
        )
        if context:
            system = f"{system}\n\n{context}"

    return [{"role": "system", "content": system}, *st.session_state.chat_messages]


def _stream_answer() -> None:
    """Генерирует ответ ассистента и дописывает его в историю."""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        collected = ""
        try:
            for piece in ollama_client.chat_stream(_build_chat_messages()):
                collected += piece
                placeholder.markdown(collected + "▌")
            placeholder.markdown(collected)
        except ollama_client.OllamaError as exc:
            placeholder.empty()
            ui.note(str(exc), kind="warn")
            return

    if collected:
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": collected}
        )


def render_chat_tab() -> None:
    """Вкладка «ИИ-ассистент» — диалог с локальной моделью."""
    status = ollama_client.check_status()
    if not status.model_ready:
        ui.note(status.message.replace("\n", "<br>"), kind="warn")
        return

    if st.session_state.scan_result is not None:
        ui.note("Результаты вашего скана переданы ассистенту.")
    else:
        ui.note("Задайте вопрос о самочувствии. Скан сделаете позже.")

    # Быстрые вопросы — экономят набор текста на телефоне.
    if not st.session_state.chat_messages:
        for i, question in enumerate(prompts.QUICK_QUESTIONS):
            if st.button(question, key=f"quick_{i}"):
                st.session_state.chat_messages.append(
                    {"role": "user", "content": question}
                )
                st.rerun()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Последнее слово за пользователем — значит, ответ ещё не сгенерирован.
    if (
        st.session_state.chat_messages
        and st.session_state.chat_messages[-1]["role"] == "user"
    ):
        _stream_answer()

    if question := st.chat_input("Что вас беспокоит?"):
        st.session_state.chat_messages.append({"role": "user", "content": question})
        st.rerun()

    if st.session_state.chat_messages and st.button("Очистить диалог", key="clear_chat"):
        st.session_state.chat_messages = []
        st.rerun()

    ui.disclaimer()


# ─────────────────────────────────────────────────────────────
# Вкладка «Куда идти»
# ─────────────────────────────────────────────────────────────


def _parse_triage(raw: str) -> dict[str, str]:
    """Разбирает ответ модели по строкам формата «КЛЮЧ: значение».

    Модель на 4B иногда добавляет вступление или меняет порядок строк,
    поэтому ищем ключи по всему тексту, а не по позициям.
    """
    fields = {
        "СРОЧНОСТЬ": "",
        "ВРАЧ": "",
        "ПОЧЕМУ": "",
        "ЧТО ДЕЛАТЬ": "",
        "ОБРАТИТЬСЯ РАНЬШЕ, ЕСЛИ": "",
    }
    for line in raw.splitlines():
        for key in fields:
            prefix = f"{key}:"
            if line.strip().upper().startswith(prefix):
                fields[key] = line.split(":", 1)[1].strip()
    return fields


def render_triage_tab() -> None:
    """Вкладка «Куда идти» — маршрутизация к специалисту."""
    status = ollama_client.check_status()
    if not status.model_ready:
        ui.note(status.message.replace("\n", "<br>"), kind="warn")
        return

    ui.note(
        "Опишите, что беспокоит — подскажем, нужен ли врач, "
        "какой специальности и насколько срочно."
    )

    complaint = st.text_area(
        "Что беспокоит",
        key="triage_input",
        height=110,
        placeholder="Например: голова болит четвёртый день, к вечеру сильнее",
        label_visibility="collapsed",
    )

    if st.button("Определить маршрут", type="primary", key="triage_run"):
        if not complaint.strip():
            ui.note("Опишите жалобы хотя бы в двух словах.", kind="warn")
        else:
            context = ""
            result = st.session_state.scan_result
            if result is not None:
                context = prompts.format_scan_context(
                    result.heart_rate, result.fatigue, result.quality
                )

            with st.spinner("Думаю…"):
                try:
                    raw = ollama_client.chat_once(
                        [
                            {"role": "system", "content": prompts.SYSTEM_TRIAGE},
                            {
                                "role": "user",
                                "content": f"{complaint}\n\n{context}".strip(),
                            },
                        ]
                    )
                except ollama_client.OllamaError as exc:
                    ui.note(str(exc), kind="warn")
                    raw = ""

            if raw:
                st.session_state.triage_result = _parse_triage(raw)

    triage = st.session_state.triage_result
    if triage:
        ui.triage_card(
            urgency=triage["СРОЧНОСТЬ"] or "ПЛАНОВО",
            doctor=triage["ВРАЧ"] or "—",
            why=triage["ПОЧЕМУ"] or "Требуется очная оценка врача.",
            what=triage["ЧТО ДЕЛАТЬ"] or "Запишитесь на приём.",
            earlier=triage["ОБРАТИТЬСЯ РАНЬШЕ, ЕСЛИ"]
            or "состояние ухудшится или появятся новые симптомы",
        )

    ui.disclaimer()


# ─────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title=config.APP_NAME,
        page_icon="🩺",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    init_session_state()
    render_header()

    tab_scan, tab_chat, tab_triage = st.tabs(["Скан", "Ассистент", "Куда идти"])

    with tab_scan:
        render_scan_tab()
    with tab_chat:
        render_chat_tab()
    with tab_triage:
        render_triage_tab()


if __name__ == "__main__":
    main()
