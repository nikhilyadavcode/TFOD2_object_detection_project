from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class DetectionResult:
    boxes: np.ndarray
    classes: np.ndarray
    scores: np.ndarray
    count: int

    def as_records(self, labels: dict[int, str], min_score: float) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for box, class_id, score in zip(self.boxes, self.classes, self.scores):
            if float(score) < min_score:
                continue
            records.append(
                {
                    "label": labels.get(int(class_id), f"class_{int(class_id)}"),
                    "class_id": int(class_id),
                    "score": round(float(score), 4),
                    "box_ymin_xmin_ymax_xmax": [round(float(value), 6) for value in box],
                }
            )
        return records


class TFOD2Detector:
    def __init__(self, saved_model_dir: Path):
        if not (saved_model_dir / "saved_model.pb").exists():
            raise FileNotFoundError(f"SavedModel not found: {saved_model_dir}")

        self.model = tf.saved_model.load(str(saved_model_dir))
        self.detect_fn = self.model.signatures["serving_default"]

    def detect(self, image_rgb: np.ndarray) -> DetectionResult:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape [height, width, 3]")

        input_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)[tf.newaxis, ...]
        detections = self.detect_fn(input_tensor)
        count = int(detections.pop("num_detections")[0].numpy())

        boxes = detections["detection_boxes"][0, :count].numpy()
        classes = detections["detection_classes"][0, :count].numpy().astype(np.int32)
        scores = detections["detection_scores"][0, :count].numpy()
        return DetectionResult(boxes=boxes, classes=classes, scores=scores, count=count)
