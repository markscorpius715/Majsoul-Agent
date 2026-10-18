from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..calibration.frames import read_image, write_image
from ..domain import Box
from .config import VisionConfig
from .schemas import GlyphPrediction


def normalize_glyph(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Trim and center a binary glyph while preserving its aspect ratio."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
        raise ValueError("Glyph mask must be a non-empty grayscale image")
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    points = cv2.findNonZero(binary)
    if points is None:
        raise ValueError("Glyph mask has no foreground")
    x, y, width, height = cv2.boundingRect(points)
    glyph = binary[y:y + height, x:x + width]
    target_width, target_height = size
    padding = 2
    scale = min((target_width - 2 * padding) / width,
                (target_height - 2 * padding) / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(glyph, (resized_width, resized_height),
                         interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width), dtype=np.uint8)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def glyph_descriptor(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    normalized = normalize_glyph(mask, size)
    pixels = normalized.astype(np.float32).reshape(-1) / 255
    # Horizontal and vertical ink projections stabilize small anti-aliasing changes.
    projections = np.concatenate((
        normalized.mean(axis=0).astype(np.float32) / 255,
        normalized.mean(axis=1).astype(np.float32) / 255,
    ))
    descriptor = np.concatenate((pixels, projections))
    norm = float(np.linalg.norm(descriptor))
    return descriptor if norm == 0 else descriptor / norm


class GlyphTemplateClassifier:
    """Classify a segmented fixed-font glyph using labeled real UI templates."""

    def __init__(self, templates: str | Path, config: VisionConfig):
        self.root = Path(templates)
        self.config = config
        self._templates: dict[str, list[np.ndarray]] = defaultdict(list)
        if self.root.exists():
            for path in sorted(self.root.glob("*/*")):
                if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                    continue
                label = path.parent.name
                if not label or any(character in label for character in "/\\"):
                    raise ValueError(f"Invalid glyph label: {label}")
                image = read_image(path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self._templates[label].append(glyph_descriptor(
                    np.where(gray >= 128, 255, 0).astype(np.uint8),
                    config.glyph_normalized_size))

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self._templates))

    def classify(self, mask: np.ndarray, box: Box, *,
                 allowed_labels=None) -> GlyphPrediction:
        if not self._templates:
            return GlyphPrediction(None, 0.0, None, None, box)
        descriptor = glyph_descriptor(mask, self.config.glyph_normalized_size)
        ordered = []
        for label, templates in self._templates.items():
            if allowed_labels is not None and label not in allowed_labels:
                continue
            distance = min(max(0.0, min(2.0, 1 - float(np.dot(descriptor, template))))
                           for template in templates)
            ordered.append((distance, label))
        if not ordered:
            return GlyphPrediction(None, 0.0, None, None, box)
        ordered.sort()
        best_distance, best_label = ordered[0]
        second_distance = ordered[1][0] if len(ordered) > 1 else 1.0
        margin = second_distance - best_distance
        accepted = (best_distance <= self.config.glyph_maximum_distance and
                    (len(ordered) == 1 or margin >= self.config.glyph_minimum_margin))
        confidence = max(0.0, min(1.0, 1 - best_distance)) if accepted else 0.0
        return GlyphPrediction(best_label if accepted else None, confidence,
                               best_distance, margin, box)


def save_glyph_template(root: str | Path, label: str, name: str, mask: np.ndarray) -> Path:
    if not label or Path(label).name != label:
        raise ValueError("Glyph label must be one directory name")
    if not name or Path(name).name != name:
        raise ValueError("Template name must be one filename")
    target = Path(root) / label / (name if Path(name).suffix else name + ".png")
    write_image(target, normalize_glyph(mask, (32, 48)))
    return target
