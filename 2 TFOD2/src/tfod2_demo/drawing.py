from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


Box = tuple[float, float, float, float]


def draw_detections(
    image: np.ndarray,
    boxes: Iterable[Box],
    classes: Iterable[int],
    scores: Iterable[float],
    labels: dict[int, str],
    min_score: float,
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]

    for box, class_id, score in zip(boxes, classes, scores):
        if score < min_score:
            continue

        ymin, xmin, ymax, xmax = box
        left = max(0, int(xmin * width))
        top = max(0, int(ymin * height))
        right = min(width - 1, int(xmax * width))
        bottom = min(height - 1, int(ymax * height))

        color = _color_for_class(class_id)
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)

        name = labels.get(class_id, f"class_{class_id}")
        caption = f"{name}: {score:.2f}"
        _draw_caption(output, caption, left, top, color)

    return output


def _draw_caption(image: np.ndarray, text: str, left: int, top: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    caption_top = max(0, top - text_height - baseline - 6)
    caption_bottom = caption_top + text_height + baseline + 6
    cv2.rectangle(image, (left, caption_top), (left + text_width + 8, caption_bottom), color, -1)
    cv2.putText(
        image,
        text,
        (left + 4, caption_bottom - baseline - 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = (
        (32, 127, 255),
        (80, 180, 80),
        (210, 90, 70),
        (180, 80, 180),
        (60, 170, 200),
        (230, 160, 40),
    )
    return palette[class_id % len(palette)]

