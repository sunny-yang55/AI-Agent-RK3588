"""Feature extraction shared by workbench shape training and runtime inference.

The white-board setup makes saturated coloured pixels a much stronger signal
than a general-purpose detector.  This module deliberately extracts only the
largest saturated object contour; callers must reject scenes with zero or
multiple target objects during labelled sample capture.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


SHAPE_LABELS = ("cube", "cylinder", "triangular_pyramid")


@dataclass(frozen=True)
class ShapeFeatures:
    values: np.ndarray
    area: float
    vertices: int


def extract_hog_shape_features(image: np.ndarray) -> np.ndarray:
    """Return image-based shape features from a padded single-block crop.

    HOG preserves the object's edges, shading and visible side faces.  Those
    cues distinguish a cube from a cylinder when simple contour statistics
    change under perspective or lighting.
    """
    import cv2

    if image is None or image.size == 0:
        raise ValueError("empty shape sample")
    height, width = image.shape[:2]
    side = max(height, width)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top : top + height, left : left + width] = image
    gray = cv2.cvtColor(cv2.resize(canvas, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((96, 96), (32, 32), (16, 16), (16, 16), 9)
    return hog.compute(gray).reshape(-1).astype(np.float32)


def extract_shape_vector(image: np.ndarray, feature_mode: str) -> np.ndarray:
    """Build the exact feature vector persisted with an SVM shape model."""
    if feature_mode == "contour":
        return extract_shape_features(image).values
    hog = extract_hog_shape_features(image)
    if feature_mode == "hog":
        return hog
    if feature_mode == "hybrid":
        return np.concatenate((extract_shape_features(image).values, hog)).astype(np.float32)
    raise ValueError(f"unsupported shape feature mode: {feature_mode}")


class WorkbenchShapeClassifier:
    """Optional local SVM classifier; absence of a reviewed model is safe."""

    def __init__(self, model_path: str | Path) -> None:
        import cv2

        self.model_path = Path(model_path)
        metadata_path = self.model_path.with_suffix(".json")
        if not self.model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"shape model is unavailable: {self.model_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.labels = tuple(metadata["labels"])
        self.feature_mode = metadata.get("feature_mode", "hybrid")
        self.mean = np.asarray(metadata["mean"], dtype=np.float32)
        self.scale = np.maximum(np.asarray(metadata["scale"], dtype=np.float32), 1e-6)
        self._model = cv2.ml.SVM_load(str(self.model_path))

    def predict(self, image: np.ndarray) -> str:
        vector = extract_shape_vector(image, self.feature_mode)
        normalized = ((vector - self.mean) / self.scale).reshape(1, -1).astype(np.float32)
        _ok, result = self._model.predict(normalized)
        index = int(result.reshape(-1)[0])
        return self.labels[index]


def extract_shape_features(image: np.ndarray) -> ShapeFeatures:
    """Return rotation-tolerant contour features for one coloured object ROI."""
    import cv2

    if image is None or image.size == 0:
        raise ValueError("empty shape sample")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # White board and black marker circles have low saturation; all four block
    # colours remain well above this threshold under the current lighting.
    mask = cv2.inRange(hsv, np.asarray((0, 55, 35)), np.asarray((180, 255, 255)))
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no saturated object contour found")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 100.0:
        raise ValueError("object contour is too small")
    perimeter = max(1.0, float(cv2.arcLength(contour, True)))
    x, y, width, height = cv2.boundingRect(contour)
    rectangle = cv2.minAreaRect(contour)
    rect_width, rect_height = rectangle[1]
    hull = cv2.convexHull(contour)
    hull_area = max(1.0, float(cv2.contourArea(hull)))
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-12))
    vertices = len(cv2.approxPolyDP(contour, 0.035 * perimeter, True))
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    image_area = max(1.0, float(image.shape[0] * image.shape[1]))
    values = np.asarray(
        [
            area / image_area,
            perimeter / max(1.0, np.sqrt(image_area)),
            circularity,
            area / max(1.0, float(width * height)),
            area / max(1.0, float(rect_width * rect_height)),
            area / hull_area,
            min(rect_width, rect_height) / max(1.0, max(rect_width, rect_height)),
            vertices / 12.0,
            *hu.tolist(),
        ],
        dtype=np.float32,
    )
    return ShapeFeatures(values=values, area=area, vertices=vertices)
