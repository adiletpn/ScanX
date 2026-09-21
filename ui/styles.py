"""Тёмная неоновая тема и мобильная вёрстка.

Демо идёт с телефона, поэтому вся вёрстка рассчитана на узкий экран:
крупные цифры, кнопки во всю ширину, ничего не уезжает вбок.
Шрифты только системные — приложение работает офлайн, Google Fonts недоступны.
"""

from __future__ import annotations

import streamlit as st

import config

#: Системный стек шрифтов: одинаково выглядит на macOS и в iOS Safari.
_FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", '
    "Roboto, Helvetica, Arial, sans-serif"
)

_CSS = f"""
<style>
:root {{
    --bg: {config.COLOR_BG};
    --cyan: {config.COLOR_CYAN};
    --magenta: {config.COLOR_MAGENTA};
    --green: {config.COLOR_GREEN};
    --amber: {config.COLOR_AMBER};
    --red: {config.COLOR_RED};
    --text: {config.COLOR_TEXT};
    --text-dim: {config.COLOR_TEXT_DIM};
    --card-bg: rgba(0, 240, 255, 0.04);
    --card-border: rgba(0, 240, 255, 0.22);
}}

/* ── Фон и базовая типографика ─────────────────────────── */
.stApp {{
    background:
        radial-gradient(ellipse 120% 80% at 50% -10%, rgba(0,240,255,0.10), transparent 60%),
        radial-gradient(ellipse 100% 60% at 90% 110%, rgba(255,43,214,0.08), transparent 60%),
        var(--bg);
    color: var(--text);
    font-family: {_FONT_STACK};
}}

/* Узкая колонка под телефон, боковые отступы 16px */
.block-container {{
    padding: 0.8rem 1rem 4rem 1rem !important;
    max-width: 560px;
}}

h1, h2, h3, h4 {{ color: var(--text); letter-spacing: -0.01em; }}

/* Streamlit прячем: меню, футер, хедер — на телефоне это мусор */
#MainMenu, footer, header {{ visibility: hidden; height: 0; }}

/* ── Шапка приложения ──────────────────────────────────── */
.sx-header {{ text-align: center; padding: 0.2rem 0 0.9rem 0; }}

.sx-logo {{
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    background: linear-gradient(92deg, var(--cyan), var(--magenta));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}}

.sx-tagline {{
    color: var(--text-dim);
    font-size: 0.82rem;
    margin-top: 0.15rem;
    letter-spacing: 0.02em;
}}

.sx-disclaimer {{
    display: inline-block;
    margin-top: 0.6rem;
    padding: 0.32rem 0.8rem;
    border: 1px solid rgba(255,176,32,0.45);
    border-radius: 999px;
    background: rgba(255,176,32,0.09);
    color: var(--amber);
    font-size: 0.7rem;
    font-weight: 600;
    line-height: 1.3;
}}

/* ── Вкладки: во всю ширину, крупные ───────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.3rem;
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 0.28rem;
    border: 1px solid rgba(255,255,255,0.06);
}}

.stTabs [data-baseweb="tab"] {{
    flex: 1;
    justify-content: center;
    height: 44px;
    border-radius: 11px;
    color: var(--text-dim);
    font-weight: 600;
    font-size: 0.86rem;
    padding: 0 0.4rem;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(0,240,255,0.16), rgba(255,43,214,0.13));
    color: var(--cyan) !important;
    box-shadow: 0 0 16px rgba(0,240,255,0.2);
}}

.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{
    display: none;
}}

/* ── Кнопки: во всю ширину, палец попадает ─────────────── */
.stButton > button, .stDownloadButton > button {{
    width: 100%;
    min-height: 50px;
    border-radius: 13px;
    border: 1px solid var(--card-border);
    background: linear-gradient(135deg, rgba(0,240,255,0.13), rgba(255,43,214,0.10));
    color: var(--text);
    font-weight: 700;
    font-size: 1rem;
    transition: box-shadow 0.18s ease, transform 0.08s ease;
}}

.stButton > button:hover {{
    border-color: var(--cyan);
    box-shadow: 0 0 22px rgba(0,240,255,0.32);
    color: var(--cyan);
}}

.stButton > button:active {{ transform: scale(0.985); }}

/* Главная кнопка действия */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, var(--cyan), #00b4d8);
    color: #02131a;
    border: none;
    box-shadow: 0 0 26px rgba(0,240,255,0.42);
}}

/* ── Карточки ──────────────────────────────────────────── */
.sx-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin: 0.55rem 0;
    backdrop-filter: blur(8px);
}}

/* Крупная метрика: пульс, усталость */
.sx-metric {{ text-align: center; padding: 0.9rem 0.5rem; }}

.sx-metric-label {{
    color: var(--text-dim);
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
}}

.sx-metric-value {{
    font-size: 3.1rem;
    font-weight: 800;
    line-height: 1.05;
    margin: 0.2rem 0;
    font-variant-numeric: tabular-nums;
}}

.sx-metric-unit {{ font-size: 1rem; font-weight: 600; color: var(--text-dim); }}
.sx-metric-note {{ font-size: 0.78rem; color: var(--text-dim); margin-top: 0.2rem; }}

.sx-glow-cyan {{ color: var(--cyan); text-shadow: 0 0 18px rgba(0,240,255,0.55); }}
.sx-glow-green {{ color: var(--green); text-shadow: 0 0 18px rgba(57,255,20,0.5); }}
.sx-glow-amber {{ color: var(--amber); text-shadow: 0 0 18px rgba(255,176,32,0.5); }}
.sx-glow-red {{ color: var(--red); text-shadow: 0 0 18px rgba(255,59,92,0.5); }}
.sx-glow-magenta {{ color: var(--magenta); text-shadow: 0 0 18px rgba(255,43,214,0.5); }}

/* ── Информационные плашки ─────────────────────────────── */
.sx-note {{
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
    margin: 0.5rem 0;
    font-size: 0.86rem;
    line-height: 1.5;
    border-left: 3px solid var(--cyan);
    background: rgba(0,240,255,0.06);
}}

.sx-note-warn {{ border-left-color: var(--amber); background: rgba(255,176,32,0.07); }}
.sx-note-danger {{ border-left-color: var(--red); background: rgba(255,59,92,0.09); }}

/* ── Поля ввода и загрузчик файлов ─────────────────────── */
[data-testid="stFileUploader"] {{
    background: rgba(255,255,255,0.025);
    border: 1.5px dashed var(--card-border);
    border-radius: 14px;
    padding: 0.5rem;
}}

[data-testid="stFileUploader"] section {{ padding: 0.6rem; }}

.stChatInput textarea, .stTextInput input, .stTextArea textarea {{
    background: rgba(255,255,255,0.05) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
    border: 1px solid var(--card-border) !important;
    font-size: 16px !important;  /* iOS Safari зумит поле, если шрифт < 16px */
}}

/* ── Чат ───────────────────────────────────────────────── */
[data-testid="stChatMessage"] {{
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.5rem;
}}

/* ── Прогресс ──────────────────────────────────────────── */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, var(--cyan), var(--magenta));
}}

/* ── Таблицы: не вылезают за экран ─────────────────────── */
[data-testid="stTable"], .stDataFrame {{ font-size: 0.82rem; }}

/* ── Совсем узкие экраны ───────────────────────────────── */
@media (max-width: 420px) {{
    .sx-logo {{ font-size: 1.8rem; }}
    .sx-metric-value {{ font-size: 2.6rem; }}
    .stTabs [data-baseweb="tab"] {{ font-size: 0.78rem; height: 42px; }}
    .block-container {{ padding: 0.6rem 0.85rem 3.5rem 0.85rem !important; }}
}}

/* Горизонтальной прокрутки страницы быть не должно */
html, body {{ overflow-x: hidden; }}
</style>
"""


def inject_styles() -> None:
    """Подключает CSS темы. Вызывать один раз в начале app.py."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_header() -> None:
    """Шапка: название, слоган и обязательный дисклеймер."""
    st.markdown(
        f"""
        <div class="sx-header">
            <div class="sx-logo">{config.APP_NAME}</div>
            <div class="sx-tagline">{config.APP_TAGLINE}</div>
            <div class="sx-disclaimer">⚠️ {config.DISCLAIMER_SHORT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
