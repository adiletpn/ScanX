"""Поиск лица, выделение ROI и сбор признаков по кадрам.

Модуль отвечает за превращение видеопотока в два набора данных:

* ряд средних R, G, B по коже лба и щёк — из него ``rppg`` достаёт пульс;
* ландмарки глаз, рта и опорных точек — из них ``fatigue`` считает
  моргания, PERCLOS и зевки.

Почему ROI именно лоб и щёки: там тонкая кожа и хорошее кровоснабжение,
а мимика и речь почти не смещают эти участки. Глаза, брови и губы
исключены намеренно — они двигаются и дают ложный сигнал.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
import numpy.typing as npt

import config

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BGRFrame = npt.NDArray[np.uint8]


class NoFaceError(RuntimeError):
    """Лицо не найдено ни на одном кадре."""


@dataclass
class ScanFeatures:
    """Признаки, собранные со всей записи."""

    timestamps: FloatArray
    """Метки времени кадров, где лицо было найдено, секунды."""

    rgb: FloatArray
    """Средние (R, G, B) по ROI, форма (N, 3), диапазон 0–255."""

    landmarks: FloatArray = field(repr=False)
    """Ландмарки в пикселях, форма (N, K, 2)."""

    face_sizes: FloatArray = field(repr=False)
    """Диагональ лица в пикселях по кадрам — для нормировки движения."""

    face_centers: FloatArray = field(repr=False)
    """Центры лица (x, y) по кадрам."""

    frames_total: int
    """Сколько кадров всего просмотрено."""

    frames_with_face: int
    """На скольких лицо найдено."""

    preview: BGRFrame | None = field(default=None, repr=False)
    """Кадр с отрисованными ROI — показываем пользователю."""

    @property
    def face_ratio(self) -> float:
        """Доля кадров с найденным лицом."""
        return self.frames_with_face / self.frames_total if self.frames_total else 0.0


def _landmarks_to_pixels(landmarks, width: int, height: int) -> FloatArray:
    """Переводит нормированные координаты Face Mesh в пиксели."""
    return np.array(
        [(lm.x * width, lm.y * height) for lm in landmarks.landmark],
        dtype=np.float64,
    )


def _roi_mask(
    points: FloatArray,
    indices: list[int],
    shape: tuple[int, int],
    shrink_px: int,
) -> npt.NDArray[np.uint8]:
    """Строит бинарную маску области по списку индексов ландмарок.

    Берём выпуклую оболочку точек, а не полигон в порядке перечисления:
    так область не выворачивается наизнанку, если индексы заданы не по
    контуру. Это делает набор ROI в config устойчивым к правкам.

    Оболочку затем сжимаем внутрь: у края лица резкий перепад яркости,
    туда же попадают волосы — и то и другое портит сигнал сильнее,
    чем помогает лишняя площадь.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)

    valid = [i for i in indices if i < len(points)]
    if len(valid) < 3:
        return mask

    hull = cv2.convexHull(points[valid].astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    if shrink_px > 0:
        kernel = np.ones((shrink_px * 2 + 1, shrink_px * 2 + 1), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def _draw_preview(frame: BGRFrame, points: FloatArray, mask: npt.NDArray[np.uint8]) -> BGRFrame:
    """Рисует кадр «вот что анализировал ИИ»: неоновые ROI и сетка точек."""
    preview = frame.copy()

    # Подсветка ROI бирюзовым.
    overlay = preview.copy()
    overlay[mask > 0] = (255, 240, 0)  # BGR: неоновый циан
    preview = cv2.addWeighted(overlay, 0.22, preview, 0.78, 0)

    # Контур областей.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (255, 240, 0), 1, cv2.LINE_AA)

    # Разреженная сетка: все 468 точек превращаются в кашу, каждая девятая
    # читается как сетка и выглядит аккуратно.
    for x, y in points[::9]:
        cv2.circle(preview, (int(x), int(y)), 1, (214, 43, 255), -1, cv2.LINE_AA)

    return preview


class FaceTracker:
    """Обёртка над MediaPipe Face Mesh со сглаживанием ландмарок.

    Детектор слегка дрожит от кадра к кадру. Без сглаживания это дрожание
    попадает прямо в сигнал ROI и маскирует пульс, амплитуда которого
    и так составляет около процента.

    Сглаживаем экспоненциально (EMA): коэффициент подобран так, чтобы убрать
    дрожание, но не «замылить» моргания — они длятся всего 3–5 кадров.
    """

    def __init__(self, alpha: float = config.LANDMARK_EMA_ALPHA) -> None:
        import mediapipe as mp  # импорт внутри: mediapipe стартует ~2 секунды

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.FACE_MESH_MAX_FACES,
            refine_landmarks=config.FACE_MESH_REFINE_LANDMARKS,
            min_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONFIDENCE,
        )
        self._alpha = alpha
        self._smoothed: FloatArray | None = None

    def __enter__(self) -> "FaceTracker":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._mesh.close()

    def process(self, frame_bgr: BGRFrame) -> FloatArray | None:
        """Находит лицо и возвращает сглаженные ландмарки в пикселях.

        Returns:
            Массив (K, 2) или None, если лицо не найдено.
        """
        height, width = frame_bgr.shape[:2]

        # MediaPipe ждёт RGB. Пометка writeable=False позволяет ему
        # работать без копии массива.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        points = _landmarks_to_pixels(result.multi_face_landmarks[0], width, height)

        if self._smoothed is None or self._smoothed.shape != points.shape:
            self._smoothed = points
        else:
            self._smoothed = (
                self._alpha * self._smoothed + (1.0 - self._alpha) * points
            )

        return self._smoothed.copy()


def extract_features(
    frames,
    progress_callback=None,
) -> ScanFeatures:
    """Прогоняет кадры через Face Mesh и собирает признаки.

    Face Mesh запускается не на каждом кадре — это самая дорогая часть
    конвейера, а между соседними кадрами лицо почти не сдвигается.
    Средние RGB при этом считаются на КАЖДОМ кадре: пропуск кадров
    в пульсовом ряду напрямую бьёт по оценке частоты.

    Args:
        frames: итератор объектов ``video_io.Frame``.
        progress_callback: необязательный вызов ``(обработано, всего)``
            для прогресс-бара.

    Returns:
        ScanFeatures с рядами RGB, ландмарками и кадром-превью.

    Raises:
        NoFaceError: лицо не найдено ни разу.
    """
    timestamps: list[float] = []
    rgb_means: list[tuple[float, float, float]] = []
    landmark_list: list[FloatArray] = []
    sizes: list[float] = []
    centers: list[tuple[float, float]] = []

    preview: BGRFrame | None = None
    frames_total = 0
    last_points: FloatArray | None = None
    last_mask: npt.NDArray[np.uint8] | None = None

    roi_indices = (
        config.ROI_FOREHEAD + config.ROI_LEFT_CHEEK + config.ROI_RIGHT_CHEEK
    )

    with FaceTracker() as tracker:
        for frame in frames:
            frames_total += 1
            image = frame.image
            height, width = image.shape[:2]

            # Прореживание детектора: на пропущенных кадрах берём маску
            # с предыдущего — лицо за 1/30 секунды не убегает.
            run_detector = (
                frame.index % config.FACE_MESH_EVERY_N == 0 or last_points is None
            )

            if run_detector:
                points = tracker.process(image)
                if points is not None:
                    last_points = points

                    # Сжатие ROI задаём в долях размера лица, чтобы оно
                    # не зависело от того, близко человек к камере или далеко.
                    face_diag = float(
                        np.linalg.norm(
                            points[config.FACE_ANCHOR_TOP]
                            - points[config.FACE_ANCHOR_BOTTOM]
                        )
                    )
                    shrink = max(int(face_diag * config.ROI_SHRINK_RATIO), 1)
                    last_mask = _roi_mask(
                        points, roi_indices, (height, width), shrink
                    )

            if last_points is None or last_mask is None or not last_mask.any():
                continue

            mean_bgr = cv2.mean(image, mask=last_mask)[:3]
            rgb_means.append((mean_bgr[2], mean_bgr[1], mean_bgr[0]))
            timestamps.append(frame.timestamp_sec)
            landmark_list.append(last_points)

            top = last_points[config.FACE_ANCHOR_TOP]
            bottom = last_points[config.FACE_ANCHOR_BOTTOM]
            left = last_points[config.FACE_ANCHOR_LEFT]
            right = last_points[config.FACE_ANCHOR_RIGHT]

            sizes.append(float(np.linalg.norm(top - bottom)))
            centers.append(
                (
                    float((left[0] + right[0]) / 2.0),
                    float((top[1] + bottom[1]) / 2.0),
                )
            )

            # Превью берём из середины записи: к этому моменту человек
            # уже сидит ровно, а до конца записи ещё далеко.
            if preview is None and frames_total > 30:
                preview = _draw_preview(image, last_points, last_mask)

            if progress_callback is not None:
                progress_callback(frames_total, None)

    if not rgb_means:
        raise NoFaceError(
            "Лицо не найдено. Снимите так, чтобы лицо было в кадре "
            "и хорошо освещено спереди."
        )

    if preview is None and last_points is not None and last_mask is not None:
        preview = _draw_preview(image, last_points, last_mask)

    log.info(
        "Признаки собраны: %d кадров с лицом из %d (%.0f%%)",
        len(rgb_means), frames_total, 100.0 * len(rgb_means) / max(frames_total, 1),
    )

    return ScanFeatures(
        timestamps=np.asarray(timestamps, dtype=np.float64),
        rgb=np.asarray(rgb_means, dtype=np.float64),
        landmarks=np.asarray(landmark_list, dtype=np.float64),
        face_sizes=np.asarray(sizes, dtype=np.float64),
        face_centers=np.asarray(centers, dtype=np.float64),
        frames_total=frames_total,
        frames_with_face=len(rgb_means),
        preview=preview,
    )
