from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..calibration.frames import read_image, write_image
from ..tiles import tile_index
from .config import VisionConfig
from .schemas import SegmentedTile, TilePrediction


def _validate_tile_label(tile: str) -> None:
    tile_index(tile)
    if tile[0] == "0" and tile[1] not in "mps":
        raise ValueError(f"Invalid red tile: {tile}")


def _rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if abs(degrees) == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError("Only right-angle rotations are supported")


def tile_descriptor(image: np.ndarray, normalized_size: tuple[int, int]) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Tile image must be a non-empty BGR image")
    width, height = normalized_size
    normalized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((width, height), (16, 16), (8, 8), (8, 8), 9)
    gradient = hog.compute(gray).reshape(-1).astype(np.float32)
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    dark = (255 - value).astype(np.float32) / 255
    red = (((hue <= 12) | (hue >= 168)) & (saturation >= 60)).astype(np.float32)
    green = ((hue >= 30) & (hue <= 100) & (saturation >= 45)).astype(np.float32)
    color = np.concatenate([
        cv2.resize(channel, (16, 24), interpolation=cv2.INTER_AREA).reshape(-1)
        for channel in (dark, red, green)
    ]).astype(np.float32)
    descriptor = np.concatenate((gradient, color))
    norm = float(np.linalg.norm(descriptor))
    if norm == 0:
        return descriptor
    return descriptor / norm


class TileTemplateClassifier:
    """Classify normalized tile faces against real, labeled UI templates."""

    def __init__(self, templates: str | Path, config: VisionConfig):
        self.root = Path(templates)
        self.config = config
        self._templates: dict[str, list[np.ndarray]] = defaultdict(list)
        if self.root.exists():
            for path in sorted(self.root.glob("*/*")):
                if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                    continue
                label = path.parent.name
                _validate_tile_label(label)
                self._templates[label].append(tile_descriptor(read_image(path), config.normalized_size))

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self._templates))

    def classify(self, candidate: SegmentedTile) -> TilePrediction:
        if not self._templates:
            return TilePrediction(None, 0.0, None, None, candidate.orientation)
        rotations = (90, -90) if candidate.orientation == "horizontal" else (0,)
        scored = []
        for rotation in rotations:
            descriptor = tile_descriptor(_rotate(candidate.image, rotation), self.config.normalized_size)
            for label, templates in self._templates.items():
                distance = min(max(0.0, min(2.0, 1 - float(np.dot(descriptor, template))))
                               for template in templates)
                scored.append((distance, label, rotation))
        best_by_label = {}
        for distance, label, rotation in scored:
            if label not in best_by_label or distance < best_by_label[label][0]:
                best_by_label[label] = (distance, rotation)
        ordered = sorted((distance, label, rotation)
                         for label, (distance, rotation) in best_by_label.items())
        best_distance, best_label, best_rotation = ordered[0]
        second_distance = ordered[1][0] if len(ordered) > 1 else 1.0
        margin = second_distance - best_distance
        accepted = (best_distance <= self.config.maximum_distance and
                    (len(ordered) == 1 or margin >= self.config.minimum_margin))
        confidence = max(0.0, min(1.0, 1.0 - best_distance)) if accepted else 0.0
        candidates = tuple((label, max(0.0, min(1.0, 1.0 - distance)))
                           for distance, label, _ in ordered[:4])
        orientation = (candidate.orientation if best_rotation == 0 else
                       f"{candidate.orientation}:{best_rotation:+d}")
        return TilePrediction(best_label if accepted else None, confidence,
                              best_distance, margin, orientation, candidates)

    def classify_many(self, candidates) -> tuple[TilePrediction, ...]:
        return tuple(self.classify(candidate) for candidate in candidates)


def save_template(root: str | Path, tile: str, name: str, image: np.ndarray) -> Path:
    _validate_tile_label(tile)
    if not name or Path(name).name != name:
        raise ValueError("Template name must be one filename")
    target = Path(root) / tile / (name if Path(name).suffix else name + ".png")
    write_image(target, image)
    return target
