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
from ai import council, ollama_client, prompts, specialties
from core import scan as scan_mod
from core import triage_engine
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
        "council_opinions": [],    # мнения специалистов консилиума
        "council_summary": None,   # сводное заключение консилиума
        "triage_decision": None,   # решение собственного движка
        "jump_to_chat": False,     # переход из скана в чат с контекстом
        "specialty": specialties.DEFAULT_KEY,  # у какого врача идёт приём
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

    if st.button("Обсудить с врачом", key="ask_ai"):
        st.session_state.jump_to_chat = True
        st.session_state.chat_messages.append(
            {"role": "user", "content": "Разбери мои результаты"}
        )
        st.rerun()


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
    system = prompts.build_specialist_prompt(st.session_state.specialty)

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

    current = specialties.get(st.session_state.specialty)

    chosen = st.selectbox(
        "Врач",
        options=[s.key for s in specialties.SPECIALTIES],
        index=[s.key for s in specialties.SPECIALTIES].index(current.key),
        format_func=lambda k: f"{specialties.BY_KEY[k].icon}  {specialties.BY_KEY[k].name}",
        label_visibility="collapsed",
    )
    if chosen != st.session_state.specialty:
        # Смена врача обнуляет диалог: прошлые ответы давал другой специалист,
        # и подмешивать их в новый приём некорректно.
        st.session_state.specialty = chosen
        st.session_state.chat_messages = []
        st.rerun()

    ui.note(f"<b>{current.icon} {current.name}</b><br>{current.complaints.capitalize()}.")

    if st.session_state.scan_result is not None:
        ui.note("Данные скана переданы врачу — он их учтёт.")

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

    if question := st.chat_input("Расскажите, что беспокоит"):
        st.session_state.chat_messages.append({"role": "user", "content": question})
        st.rerun()

    if st.session_state.chat_messages and st.button("Очистить диалог", key="clear_chat"):
        st.session_state.chat_messages = []
        st.rerun()


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
        "ПРИЧИНЫ": "",
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


def _run_council(complaint: str, scan_context: str) -> None:
    """Собирает консилиум: выбор врачей, мнения, общее заключение.

    Мнения стримятся по мере генерации — на локальной модели весь разбор
    занимает около минуты, и без потокового вывода экран выглядел бы
    зависшим всё это время.
    """
    st.session_state.council_opinions = []
    st.session_state.council_summary = None

    # Решение принимает наш движок — мгновенно, детерминированно
    # и с объяснением. Модель подключается уже после, только чтобы
    # изложить это человеческим языком.
    result = st.session_state.scan_result
    decision = triage_engine.evaluate(
        complaint,
        heart_rate_bpm=(
            result.heart_rate.bpm
            if result is not None and result.quality.level.is_trustworthy
            else None
        ),
        fatigue_index=(
            result.fatigue.index
            if result is not None and result.fatigue.reliable
            else None
        ),
    )
    st.session_state.triage_decision = decision
    ui.decision_trace(decision)

    # Неотложное состояние — единственный случай, когда мы не ждём модель:
    # номер скорой должен появиться на экране немедленно.
    if decision.is_emergency:
        ui.emergency_card(decision)
        return

    chosen = council.choose_specialists(decision)

    opinions: list[council.Opinion] = []
    for specialty in chosen:
        st.markdown(
            f'<div class="sx-metric-label">{specialty.icon} {specialty.name}</div>',
            unsafe_allow_html=True,
        )
        placeholder = st.empty()
        collected = ""
        try:
            for piece in council.opinion_stream(specialty, complaint, scan_context):
                collected += piece
                placeholder.markdown(collected + "▌")
            placeholder.markdown(collected)
        except ollama_client.OllamaError as exc:
            placeholder.empty()
            ui.note(str(exc), kind="warn")
            return
        opinions.append(council.Opinion(specialty=specialty, text=collected))

    st.session_state.council_opinions = opinions

    try:
        with st.spinner("Свожу заключение…"):
            raw = council.summarize(complaint, opinions)
    except ollama_client.OllamaError as exc:
        ui.note(str(exc), kind="warn")
        return

    st.session_state.council_summary = council.parse_summary(raw)


def render_triage_tab() -> None:
    """Вкладка «Куда идти» — маршрутизация к специалисту."""
    status = ollama_client.check_status()
    if not status.model_ready:
        ui.note(status.message.replace("\n", "<br>"), kind="warn")
        return

    ui.note(
        "Опишите жалобы подробно: что беспокоит, как давно, "
        "что усиливает и что облегчает. Чем больше деталей — тем точнее разбор."
    )

    complaint = st.text_area(
        "Что беспокоит",
        key="triage_input",
        height=110,
        placeholder="Например: голова болит четвёртый день, к вечеру сильнее",
        label_visibility="collapsed",
    )

    scan_context = ""
    if st.session_state.scan_result is not None:
        result = st.session_state.scan_result
        scan_context = prompts.format_scan_context(
            result.heart_rate, result.fatigue, result.quality
        )

    if st.button("Созвать консилиум", type="primary", key="council_run"):
        if not complaint.strip():
            ui.note("Опишите жалобы хотя бы в двух словах.", kind="warn")
        else:
            _run_council(complaint, scan_context)

    if st.session_state.council_summary:
        summary = st.session_state.council_summary
        ui.council_card(
            agreement=summary["СОГЛАСИЕ"],
            disagreement=summary["РАЗНОГЛАСИЯ"],
            urgency=summary["СРОЧНОСТЬ"] or "ПЛАНОВО",
            doctor=summary["ГЛАВНЫЙ ВРАЧ"],
            plan=summary["ПЛАН"],
        )

    if st.button("Быстрое заключение (один врач)", key="triage_run"):
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
        matched = specialties.match_by_name(triage["ВРАЧ"])
        if matched is not None and st.button(
            f"Записаться к специалисту: {matched.icon} {matched.name}", key="goto_spec"
        ):
            st.session_state.specialty = matched.key
            st.session_state.chat_messages = []
            st.rerun()

        ui.triage_card(
            urgency=triage["СРОЧНОСТЬ"] or "ПЛАНОВО",
            doctor=triage["ВРАЧ"] or "—",
            causes=triage["ПРИЧИНЫ"],
            why=triage["ПОЧЕМУ"] or "Требуется очная оценка врача.",
            what=triage["ЧТО ДЕЛАТЬ"] or "Запишитесь на приём.",
            earlier=triage["ОБРАТИТЬСЯ РАНЬШЕ, ЕСЛИ"]
            or "состояние ухудшится или появятся новые симптомы",
        )


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

    tab_scan, tab_chat, tab_triage = st.tabs(["Скан", "Приём", "Заключение"])

    with tab_scan:
        render_scan_tab()
    with tab_chat:
        render_chat_tab()
    with tab_triage:
        render_triage_tab()

    # Дисклеймер один на всё приложение: повторённый на каждой вкладке,
    # он превращается в шум, который перестают читать.
    st.divider()
    ui.disclaimer()


if __name__ == "__main__":
    main()
