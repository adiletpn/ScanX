"""Прогон скана из терминала — для отладки без запуска Streamlit.

Поднимать веб-интерфейс ради каждой проверки долго, а здесь виден весь
разбор сразу: сколько кадров нашлось, какой пульс по каждому окну,
где просело качество.

    python scan_cli.py demo/IMG_9857.MOV
    python scan_cli.py demo/*.MOV --preview out.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

import config
from core import scan as scan_mod


def _bar(value: float, low: float, high: float, width: int = 28) -> str:
    """Рисует положение значения на шкале нормы — быстрый визуальный контроль."""
    span = high - low
    position = int(round((value - low) / span * width)) if span > 0 else 0
    position = max(0, min(width, position))
    return "[" + "·" * position + "●" + "·" * (width - position) + "]"


def run_one(path: Path, preview_out: Path | None) -> int:
    """Прогоняет один файл и печатает разбор. Возвращает код возврата."""
    print(f"\n{'=' * 60}\n{path.name}\n{'=' * 60}")

    def progress(fraction: float, label: str) -> None:
        sys.stdout.write(f"\r  {label:<28} {fraction * 100:3.0f}%")
        sys.stdout.flush()

    try:
        result = scan_mod.run_scan(path, progress=progress)
    except scan_mod.ScanError as exc:
        print(f"\n  ОШИБКА: {exc}")
        return 1

    print("\n")

    hr = result.heart_rate
    low, high = config.HR_RESTING_NORMAL

    print(f"  ПУЛЬС          {hr.bpm:5.1f} уд/мин   {_bar(hr.bpm, 40, 140)}")
    print(f"    разброс      ±{hr.bpm_spread:.1f} (межквартильный размах по окнам)")
    print(f"    SNR          {hr.snr_db:.1f} дБ")
    print(f"    окон         {hr.n_windows}")
    print(
        f"    по окнам     мин {hr.window_bpms.min():.0f} · "
        f"медиана {np.median(hr.window_bpms):.0f} · макс {hr.window_bpms.max():.0f}"
    )
    print(f"    норма покоя  {low:.0f}–{high:.0f}")

    fat = result.fatigue
    print(f"\n  УСТАЛОСТЬ      {fat.index:5.1f} / 100  ({fat.level.label})")
    print(f"    морганий     {fat.blink_count} → {fat.blink_rate_per_min:.1f}/мин")
    print(f"    длительность {fat.mean_blink_duration_ms:.0f} мс")
    print(f"    PERCLOS      {fat.perclos * 100:.1f}%")
    print(f"    зевков       {fat.yawn_count}")
    print(f"    надёжно      {'да' if fat.reliable else 'нет'}")

    qual = result.quality
    print(f"\n  КАЧЕСТВО       {qual.level.emoji} {qual.level.label}")
    print(f"    лицо в кадре {qual.face_ratio * 100:.1f}%")
    print(f"    движение     {qual.motion:.4f}")
    print(f"    яркость ROI  {qual.brightness:.0f} / 255")
    for hint in qual.hints:
        print(f"    • {hint}")

    print(
        f"\n  ВРЕМЯ          обработка {result.processing_sec:.1f} с "
        f"на запись {result.duration_sec:.1f} с"
    )
    if result.processing_sec > 30:
        print("    ⚠ дольше 30 с — поднимите FACE_MESH_EVERY_N в config.py")

    if not qual.level.is_trustworthy:
        print("\n  ⚠ Качество низкое — пульс пользователю показан не будет.")

    if preview_out is not None and result.preview is not None:
        import cv2

        cv2.imwrite(str(preview_out), result.preview)
        print(f"\n  Превью ROI сохранено: {preview_out}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогон скана ScanX из терминала")
    parser.add_argument("videos", nargs="+", type=Path, help="видеофайлы")
    parser.add_argument(
        "--preview", type=Path, default=None, help="куда сохранить кадр с ROI"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    failures = 0
    for path in args.videos:
        if not path.exists():
            print(f"нет файла: {path}")
            failures += 1
            continue
        failures += run_one(path, args.preview)

    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
