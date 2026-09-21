"""Готовые блоки интерфейса: карточки метрик, графики, результаты.

Всё рассчитано на экран телефона: одна колонка, крупные цифры,
никаких таблиц, которые уезжают вбок.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from core.fatigue import FatigueResult
from core.quality import QualityLevel, QualityReport
from core.rppg import HeartRateResult


def metric_card(
    label: str,
    value: str,
    unit: str = "",
    note: str = "",
    glow: str = "cyan",
) -> None:
    """Крупная метрика с неоновым свечением.

    Args:
        label: подпись сверху, например «Пульс».
        value: само число.
        unit: единицы измерения.
        note: строка под числом — норма или пояснение.
        glow: цвет свечения: cyan / green / amber / red / magenta.
    """
    unit_html = f'<span class="sx-metric-unit"> {unit}</span>' if unit else ""
    note_html = f'<div class="sx-metric-note">{note}</div>' if note else ""

    st.markdown(
        f"""
        <div class="sx-card sx-metric">
            <div class="sx-metric-label">{label}</div>
            <div class="sx-metric-value sx-glow-{glow}">{value}{unit_html}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(text: str, kind: str = "info") -> None:
    """Информационная плашка. kind: info / warn / danger."""
    suffix = {"info": "", "warn": " sx-note-warn", "danger": " sx-note-danger"}[kind]
    st.markdown(f'<div class="sx-note{suffix}">{text}</div>', unsafe_allow_html=True)


def _hr_status(bpm: float) -> tuple[str, str]:
    """Сравнивает пульс с нормой покоя. Возвращает (текст, цвет свечения)."""
    low, high = config.HR_RESTING_NORMAL
    if bpm < low:
        return f"ниже обычного (норма покоя {low:.0f}–{high:.0f})", "amber"
    if bpm > high:
        return f"выше обычного (норма покоя {low:.0f}–{high:.0f})", "amber"
    return f"в пределах нормы покоя ({low:.0f}–{high:.0f})", "green"


def heart_rate_card(hr: HeartRateResult, quality: QualityReport) -> None:
    """Карточка пульса.

    При плохом качестве записи число не показывается вообще. Показать его
    с оговоркой мелким шрифтом бесполезно: цифру запомнят, а оговорку нет.
    """
    if not quality.level.is_trustworthy:
        metric_card(
            label="Пульс",
            value="—",
            note="Качество записи низкое. Нужен повторный замер.",
            glow="red",
        )
        return

    status, glow = _hr_status(hr.bpm)
    spread = f" · разброс ±{hr.bpm_spread:.0f}" if hr.bpm_spread >= 1.0 else ""

    metric_card(
        label="Пульс",
        value=f"{hr.bpm:.0f}",
        unit="уд/мин",
        note=f"{status}{spread}",
        glow=glow,
    )


def fatigue_card(fatigue: FatigueResult) -> None:
    """Карточка усталости."""
    if not fatigue.reliable:
        metric_card(
            label="Усталость",
            value="—",
            note="Запись слишком короткая для оценки.",
            glow="amber",
        )
        return

    glow = {"low": "green", "moderate": "amber", "high": "red"}[fatigue.level.value]
    details = f"{fatigue.blink_rate_per_min:.0f} морганий/мин"
    if fatigue.yawn_count:
        details += f" · зевков: {fatigue.yawn_count}"

    metric_card(
        label="Индекс усталости",
        value=f"{fatigue.index:.0f}",
        unit="/ 100",
        note=f"{fatigue.level.label} · {details}",
        glow=glow,
    )


def quality_card(quality: QualityReport) -> None:
    """Карточка качества записи с подсказками."""
    glow = {"good": "green", "fair": "amber", "poor": "red"}[quality.level.value]

    metric_card(
        label="Качество записи",
        value=quality.level.emoji,
        note=f"лицо в кадре {quality.face_ratio * 100:.0f}% · SNR {quality.snr_db:.1f} дБ",
        glow=glow,
    )

    if quality.hints:
        kind = "danger" if quality.level is QualityLevel.POOR else "warn"
        if quality.level is QualityLevel.GOOD:
            kind = "info"
        note("<br>".join(f"• {h}" for h in quality.hints), kind=kind)


def pulse_wave_chart(hr: HeartRateResult, seconds: float = 8.0) -> None:
    """График пульсовой волны.

    Рисуем не всю запись, а первые несколько секунд: на двадцати секундах
    отдельные удары сливаются в сплошную заливку и смотреть не на что.
    """
    n = min(int(seconds * config.RESAMPLE_FPS), hr.wave.size)
    if n < 10:
        return

    st.caption("Пульсовая волна")
    st.line_chart(
        {"сигнал": hr.wave[:n]},
        height=140,
        color=config.COLOR_CYAN,
    )


def spectrum_chart(hr: HeartRateResult) -> None:
    """Спектр сигнала в диапазоне пульса с подписанным пиком."""
    lo, hi = config.HR_MIN_BPM / 60.0, config.HR_MAX_BPM / 60.0
    band = (hr.freqs >= lo) & (hr.freqs <= hi)
    if not band.any():
        return

    bpm_axis = hr.freqs[band] * 60.0
    values = hr.spectrum[band]

    st.caption(f"Спектр · пик на {hr.bpm:.0f} уд/мин")

    # Индексируем по частоте в уд/мин: так подпись оси сразу читается
    # как пульс, без мысленного перевода из герц.
    frame = pd.DataFrame({"уд/мин": bpm_axis, "мощность": values}).set_index("уд/мин")
    st.line_chart(frame, height=140, color=config.COLOR_MAGENTA)


def preview_image(preview) -> None:
    """Кадр с размеченными ROI — «вот что анализировал ИИ»."""
    if preview is None:
        return
    st.caption("Области, по которым измерялся кровоток")
    # OpenCV держит кадры в BGR, Streamlit ждёт RGB.
    st.image(preview[:, :, ::-1], use_container_width=True)


def comparison_card(
    baseline_bpm: float,
    current_bpm: float,
    label_before: str = "В покое",
    label_after: str = "После нагрузки",
) -> None:
    """Сравнение двух замеров — проба с нагрузкой.

    Показывает не только два числа, но и разницу: именно она говорит
    о том, как сердце отвечает на нагрузку.
    """
    delta = current_bpm - baseline_bpm
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
    glow = "magenta" if abs(delta) >= 10 else "cyan"

    st.markdown(
        f"""
        <div class="sx-card">
            <div class="sx-metric-label" style="text-align:center">
                Проба с нагрузкой
            </div>
            <div style="display:flex; justify-content:space-around;
                        align-items:center; margin-top:0.7rem">
                <div style="text-align:center">
                    <div class="sx-metric-note">{label_before}</div>
                    <div style="font-size:1.9rem; font-weight:800"
                         class="sx-glow-cyan">{baseline_bpm:.0f}</div>
                </div>
                <div style="text-align:center">
                    <div class="sx-metric-note">разница</div>
                    <div style="font-size:1.9rem; font-weight:800"
                         class="sx-glow-{glow}">{arrow} {abs(delta):.0f}</div>
                </div>
                <div style="text-align:center">
                    <div class="sx-metric-note">{label_after}</div>
                    <div style="font-size:1.9rem; font-weight:800"
                         class="sx-glow-magenta">{current_bpm:.0f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def triage_card(
    urgency: str,
    doctor: str,
    causes: str,
    why: str,
    what: str,
    earlier: str,
) -> None:
    """Карточка заключения приёма: вердикт, причины, план действий.

    Исход «САМОСТОЯТЕЛЬНО» здесь полноценный, а не отговорка: большинство
    обращений к терапевту заканчиваются именно им, и человеку нужен
    конкретный план на дом, а не отправка в очередь на всякий случай.
    """
    palette = {
        "СРОЧНО": ("🔴", "red", "Нужна помощь сегодня"),
        "ПЛАНОВО": ("🟡", "amber", "К врачу в ближайшие дни"),
        "САМОСТОЯТЕЛЬНО": ("🟢", "green", "Врач не нужен — справитесь сами"),
    }
    emoji, glow, headline = palette.get(urgency.upper(), ("🟡", "amber", urgency))

    doctor_html = (
        f'<div style="margin-top:0.5rem"><b>Специалист:</b> {doctor}</div>'
        if doctor and doctor.lower() not in ("—", "не нужен", "")
        else ""
    )
    causes_html = (
        f'<div style="margin-top:0.5rem"><b>Вероятные причины:</b> {causes}</div>'
        if causes
        else ""
    )

    st.markdown(
        f"""
        <div class="sx-card">
            <div style="text-align:center; font-size:2.4rem">{emoji}</div>
            <div style="text-align:center; font-size:1.25rem; font-weight:800;
                        margin-bottom:0.6rem" class="sx-glow-{glow}">{headline}</div>
            <div style="font-size:0.9rem; line-height:1.55">
                {why}
                {causes_html}
                {doctor_html}
                <div style="margin-top:0.5rem"><b>План действий:</b> {what}</div>
                <div style="margin-top:0.5rem; color:{config.COLOR_AMBER}">
                    <b>Обратиться раньше, если:</b> {earlier}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def council_card(
    agreement: str,
    disagreement: str,
    urgency: str,
    doctor: str,
    plan: str,
) -> None:
    """Сводное заключение консилиума.

    Разногласия показываем отдельной строкой и не прячем: если специалисты
    разошлись во мнениях, человек должен об этом знать — это сигнал, что
    случай неоднозначный и очный приём нужен сильнее.
    """
    palette = {
        "СРОЧНО": ("🔴", "red", "Нужна помощь сегодня"),
        "ПЛАНОВО": ("🟡", "amber", "К врачу в ближайшие дни"),
        "САМОСТОЯТЕЛЬНО": ("🟢", "green", "Врач не нужен — справитесь сами"),
    }
    emoji, glow, headline = palette.get(urgency.upper(), ("🟡", "amber", urgency))

    rows = [f"<div style='margin-top:0.5rem'><b>Согласие:</b> {agreement}</div>"]
    if disagreement and disagreement.lower() not in ("нет", "—", ""):
        rows.append(
            f"<div style='margin-top:0.5rem; color:{config.COLOR_MAGENTA}'>"
            f"<b>Разногласия:</b> {disagreement}</div>"
        )
    if doctor:
        rows.append(f"<div style='margin-top:0.5rem'><b>К кому в первую очередь:</b> {doctor}</div>")
    if plan:
        rows.append(f"<div style='margin-top:0.5rem'><b>План:</b> {plan}</div>")

    st.markdown(
        f"""
        <div class="sx-card">
            <div class="sx-metric-label" style="text-align:center">
                Заключение консилиума
            </div>
            <div style="text-align:center; font-size:2.2rem; margin-top:0.4rem">{emoji}</div>
            <div style="text-align:center; font-size:1.2rem; font-weight:800;
                        margin-bottom:0.4rem" class="sx-glow-{glow}">{headline}</div>
            <div style="font-size:0.9rem; line-height:1.55">
                {"".join(rows)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    """Обязательный дисклеймер под результатами."""
    st.caption(f"⚠️ {config.DISCLAIMER_FULL}")
