"""Полный прогон обоих модулей — то, что показывается на сцене.

Запуск: python demo_run.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from core.asr import transcribe_file
from core.alerts import scam_alert
from core.scam_engine import evaluate as scam_eval, warning_text

W = 68
CALLS = {
    "New Recording.m4a": "«Служба безопасности банка»",
    "New Recording 2.m4a": "«Ваш сын попал в аварию»",
    "New Recording 4.m4a": "Казахский: «Кодты айтыңыз»",
    "New Recording 6.m4a": "Настоящий банк (не должен сработать)",
    "New Recording 7.m4a": "Сын звонит маме (не должен сработать)",
}


def rule(char: str = "─") -> None:
    print(char * W)


def bar(score: float) -> str:
    filled = min(int(score / 12 * 24), 24)
    return "█" * filled + "░" * (24 - filled)


def demo_guard() -> None:
    print()
    rule("═")
    print("  МОДУЛЬ 1 · ЗАЩИТА ОТ МОШЕННИКОВ")
    rule("═")

    folder = Path("demo/calls")
    for name, label in CALLS.items():
        path = folder / name
        if not path.exists():
            continue

        print(f"\n▶ {label}")
        t0 = time.perf_counter()
        text = transcribe_file(path, dual=True)
        dt = time.perf_counter() - t0
        decision = scam_eval(text)

        print(f"  услышано: …{text[len(text)//3:len(text)//3+72].strip()}…")
        print(f"  {decision.level.emoji} [{bar(decision.score)}] {decision.score:5.1f}"
              f"  {decision.level.label}")
        print(f"  распознано за {dt:.1f} с")

        if decision.matched:
            for pattern in decision.matched[:3]:
                print(f"     • {pattern.name}")
        if decision.safe_signals:
            for signal in decision.safe_signals[:2]:
                print(f"     ✓ {signal}")

        spoken = warning_text(decision)
        if spoken:
            print(f"\n  📢 ТЕЛЕФОН ГОВОРИТ ВСЛУХ:")
            print(f"     «{spoken}»")
            alert = scam_alert(decision, "Маме")
            print(f"\n  📩 SMS ДОЧЕРИ:")
            for line in alert.body.splitlines():
                print(f"     {line}")
        else:
            print("  🤫 система молчит — тревоги нет")


def demo_scan() -> None:
    print()
    rule("═")
    print("  МОДУЛЬ 2 · ПРОВЕРКА ЗДОРОВЬЯ")
    rule("═")

    from core import history
    from core.scan import ScanError, run_scan

    video = next(iter(sorted(Path("demo").glob("*.MOV"))), None)
    if video is None:
        print("\n  нет видео в demo/")
        return

    print(f"\n▶ {video.name}")
    t0 = time.perf_counter()
    try:
        result = run_scan(video)
    except ScanError as exc:
        print(f"  ошибка: {exc}")
        return
    dt = time.perf_counter() - t0

    print(f"  ❤️  пульс      {result.bpm_display} уд/мин")
    print(f"  😴 усталость  {result.fatigue.index:.0f} / 100 ({result.fatigue.level.label})")
    print(f"  📶 качество   {result.quality.level.emoji} {result.quality.level.label}")
    print(f"  ⏱  обработано за {dt:.0f} с (запись {result.duration_sec:.0f} с)")

    baseline = history.compute_baseline()
    deviation = history.check_deviation(result.heart_rate.bpm, baseline)
    print(f"\n  📊 {deviation.message}")


if __name__ == "__main__":
    print()
    print("  ScanX — цифровой защитник".center(W))
    print("  всё офлайн, без единого запроса в сеть".center(W))
    demo_guard()
    if "--scan" in sys.argv:
        demo_scan()
    print()
    rule("═")
    print("  Ни одного обращения в интернет за весь прогон.")
    rule("═")
    print()
