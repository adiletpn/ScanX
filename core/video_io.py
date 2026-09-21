"""Чтение видео: метки времени, поворот, масштабирование.

Три вещи, на которых обычно спотыкается обработка видео с телефона:

1. **Поворот.** Снятое вертикально видео физически хранится горизонтально,
   а ориентация лежит в метаданных. OpenCV с версии 4.5 разворачивает его
   сам, но не всегда — поэтому проверяем результат и при необходимости
   доворачиваем вручную.

2. **Плавающий FPS.** Камера подстраивает выдержку под освещение, и интервалы
   между кадрами гуляют. Заявленный в файле FPS при этом врёт. Поэтому берём
   метку времени каждого кадра (``CAP_PROP_POS_MSEC``) и работаем с реальным
   временем.

3. **HEVC с iPhone.** OpenCV читает его через раз. Если не вышло —
   перекодируем файл через ffmpeg в H.264 и пробуем снова.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import numpy.typing as npt

import config

log = logging.getLogger(__name__)

BGRFrame = npt.NDArray[np.uint8]


class VideoReadError(RuntimeError):
    """Видео не удалось прочитать.

    Отдельный тип, чтобы UI мог показать человеческое объяснение
    вместо трассировки стека.
    """


@dataclass(frozen=True)
class VideoInfo:
    """Характеристики открытого видеофайла."""

    width: int
    height: int
    frame_count: int
    declared_fps: float
    """FPS из метаданных. Для оценки пульса не используем — он врёт."""

    duration_sec: float
    rotation: int
    """Поворот из метаданных, градусы: 0, 90, 180 или 270."""


@dataclass
class Frame:
    """Один кадр видео вместе с реальной меткой времени."""

    index: int
    timestamp_sec: float
    image: BGRFrame


# ─────────────────────────────────────────────────────────────
# Служебное
# ─────────────────────────────────────────────────────────────


def _probe_rotation(path: Path) -> int:
    """Достаёт угол поворота из метаданных через ffprobe.

    OpenCV эту информацию не отдаёт, а знать её нужно: если кадр придёт
    боком, Face Mesh лицо просто не найдёт.

    Returns:
        Угол в градусах (0/90/180/270). При любой ошибке — 0.
    """
    if not shutil.which("ffprobe"):
        return 0

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream_side_data=rotation:stream_tags=rotate",
                "-of", "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        for line in result.stdout.split():
            try:
                # ffprobe отдаёт отрицательные углы: -90 это те же 270.
                return int(float(line)) % 360
            except ValueError:
                continue
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("ffprobe не смог прочитать поворот: %s", exc)

    return 0


def _apply_rotation(frame: BGRFrame, rotation: int) -> BGRFrame:
    """Доворачивает кадр на заданный угол."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _transcode_to_h264(src: Path) -> Path:
    """Перекодирует видео в H.264 — запасной путь для HEVC с iPhone.

    Заодно ffmpeg применяет поворот из метаданных, так что на выходе
    кадры уже правильно ориентированы.

    Raises:
        VideoReadError: ffmpeg недоступен или не справился.
    """
    if not shutil.which("ffmpeg"):
        raise VideoReadError(
            "Не удалось прочитать видео. Снимите в формате «Наиболее совместимые» "
            "(Настройки → Камера → Форматы) или загрузите mp4."
        )

    dst = Path(tempfile.mkdtemp(prefix="scanx_")) / "converted.mp4"
    log.info("Перекодирую %s → H.264", src.name)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(src),
                "-vf", "scale=640:-2",     # сразу ужимаем — дальше всё равно ужимать
                "-c:v", "libx264",
                "-preset", "ultrafast",    # скорость важнее размера файла
                "-an",                      # звук не нужен
                str(dst),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")[-400:]
        log.error("ffmpeg упал: %s", stderr)
        raise VideoReadError(
            "Файл повреждён или это не видео. Попробуйте записать заново."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoReadError("Обработка заняла слишком много времени.") from exc

    if not dst.exists() or dst.stat().st_size == 0:
        raise VideoReadError("Не удалось преобразовать видео.")

    return dst


def _resize_to_width(frame: BGRFrame, target_width: int) -> BGRFrame:
    """Ужимает кадр до заданной ширины, сохраняя пропорции.

    На 640 px MediaPipe находит лицо не хуже, чем на 1080p, а считает втрое
    быстрее. Апскейлить не будем: качества это не добавит.
    """
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    scale = target_width / width
    new_size = (target_width, int(round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────
# Открытие и чтение
# ─────────────────────────────────────────────────────────────


def save_upload_to_temp(data: bytes, suffix: str = ".mp4") -> Path:
    """Сохраняет загруженный файл во временный — cv2 умеет только пути."""
    if not data:
        raise VideoReadError("Файл пустой.")
    tmp_dir = Path(tempfile.mkdtemp(prefix="scanx_"))
    path = tmp_dir / f"upload{suffix}"
    path.write_bytes(data)
    return path


def probe(path: Path) -> tuple[cv2.VideoCapture, VideoInfo, int]:
    """Открывает видео и собирает его характеристики.

    При неудаче пробует перекодировать через ffmpeg — это лечит HEVC с iPhone.

    Returns:
        Кортеж (открытый VideoCapture, характеристики, угол доворота).
        Угол доворота — сколько ещё нужно повернуть кадр вручную:
        если ffmpeg уже всё сделал, там будет 0.

    Raises:
        VideoReadError: файл не открылся и после перекодирования.
    """
    path = Path(path)
    if not path.exists():
        raise VideoReadError("Файл не найден.")

    rotation = _probe_rotation(path)
    cap = cv2.VideoCapture(str(path))

    # Мало открыть файл — нужно убедиться, что кадры реально читаются.
    # С HEVC бывает так, что isOpened() истинно, а read() возвращает пусто.
    ok, probe_frame = (cap.read() if cap.isOpened() else (False, None))

    if not ok or probe_frame is None:
        cap.release()
        if not config.FFMPEG_FALLBACK_ENABLED:
            raise VideoReadError("Не удалось прочитать видео.")

        log.info("OpenCV не справился — пробую через ffmpeg")
        converted = _transcode_to_h264(path)
        cap = cv2.VideoCapture(str(converted))
        ok, probe_frame = (cap.read() if cap.isOpened() else (False, None))
        if not ok or probe_frame is None:
            cap.release()
            raise VideoReadError(
                "Не удалось прочитать видео. Попробуйте записать заново."
            )
        # ffmpeg уже развернул кадры по метаданным — вручную больше не нужно.
        rotation = 0

    height, width = probe_frame.shape[:2]

    # Если OpenCV уже развернул кадр сам, ширина будет меньше высоты
    # у портретного видео — тогда второй доворот всё сломает.
    manual_rotation = rotation
    if rotation in (90, 270) and height > width:
        manual_rotation = 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    declared_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = frame_count / declared_fps if declared_fps > 0 else 0.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # отматываем назад после пробы

    if manual_rotation in (90, 270):
        width, height = height, width

    info = VideoInfo(
        width=width,
        height=height,
        frame_count=frame_count,
        declared_fps=declared_fps,
        duration_sec=duration,
        rotation=rotation,
    )
    log.info(
        "Видео: %dx%d, %d кадров, заявлено %.1f fps, поворот %d°",
        width, height, frame_count, declared_fps, rotation,
    )
    return cap, info, manual_rotation


def iter_frames(
    path: Path,
    target_width: int = config.TARGET_WIDTH,
    max_frames: int = config.MAX_FRAMES,
) -> Iterator[Frame]:
    """Идёт по кадрам видео, отдавая каждый с реальной меткой времени.

    Генератор, а не список: 20 секунд в 640 px — это около 600 кадров,
    держать их все в памяти незачем.

    Yields:
        Frame — индекс, время в секундах от начала, повёрнутый и ужатый кадр.

    Raises:
        VideoReadError: файл не читается или в нём нет кадров.
    """
    cap, _info, manual_rotation = probe(path)

    try:
        index = 0
        last_ts = -1.0

        while index < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # Реальная метка времени кадра. Именно она, а не номер кадра,
            # делает оценку частоты корректной при плавающем FPS.
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = float(ts_ms) / 1000.0 if ts_ms and ts_ms > 0 else -1.0

            # Некоторые контейнеры отдают нули или повторы. В этом случае
            # откатываемся на заявленный FPS — хуже, но лучше, чем ничего.
            if timestamp <= last_ts:
                fps = _info.declared_fps if _info.declared_fps > 0 else 30.0
                timestamp = (last_ts + 1.0 / fps) if last_ts >= 0 else index / fps
            last_ts = timestamp

            frame = _apply_rotation(frame, manual_rotation)
            frame = _resize_to_width(frame, target_width)

            yield Frame(index=index, timestamp_sec=timestamp, image=frame)
            index += 1

        if index == 0:
            raise VideoReadError("В видео нет кадров.")

    finally:
        cap.release()


def validate_duration(timestamps: list[float]) -> None:
    """Проверяет, что записи хватает для анализа.

    Raises:
        VideoReadError: с текстом, готовым к показу пользователю.
    """
    if len(timestamps) < 2:
        raise VideoReadError("В видео слишком мало кадров.")

    duration = timestamps[-1] - timestamps[0]
    if duration < config.MIN_DURATION_SEC:
        raise VideoReadError(
            f"Видео длится {duration:.0f} с. "
            f"Запишите хотя бы {config.RECOMMENDED_DURATION_SEC} секунд."
        )
