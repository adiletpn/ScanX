"""ScanX — точка входа Streamlit.

Медицинский помощник первого шага: скан по видео лица, ИИ-ассистент
и маршрутизация к врачу. Работает полностью офлайн.

Запуск:
    streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

import config
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
        "lang": "ru",              # язык интерфейса и ответов ИИ
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_scan_tab() -> None:
    """Вкладка «Скан» — загрузка видео и результаты замера."""
    st.markdown(
        f"""
        <div class="sx-note">
            <b>Как записать</b><br>
            Лицо в кадре, ровный свет спереди.
            Не двигайтесь и не говорите {config.RECOMMENDED_DURATION_SEC} секунд.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.file_uploader(
        "Видео лица",
        type=config.ALLOWED_VIDEO_TYPES,
        key="scan_video",
        label_visibility="collapsed",
    )

    st.button("Начать анализ", type="primary", key="scan_run", disabled=True)
    st.button("Демо-видео", key="scan_demo", disabled=True)

    st.caption(config.DISCLAIMER_FULL)


def render_chat_tab() -> None:
    """Вкладка «ИИ-ассистент» — диалог с локальной моделью."""
    st.markdown(
        """
        <div class="sx-note">
            Задайте вопрос о самочувствии. Если вы уже сделали скан,
            ассистент увидит его результаты.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.chat_input("Что вас беспокоит?", disabled=True)
    st.caption(config.DISCLAIMER_FULL)


def render_triage_tab() -> None:
    """Вкладка «Куда идти» — маршрутизация к специалисту."""
    st.markdown(
        """
        <div class="sx-note">
            Опишите симптомы — подскажем, нужен ли врач,
            какой специальности и насколько срочно.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.text_area("Что беспокоит", key="triage_input", height=110, disabled=True)
    st.button("Определить маршрут", type="primary", key="triage_run", disabled=True)
    st.caption(config.DISCLAIMER_FULL)


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
