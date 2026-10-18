from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from ..calibration.frames import read_image
from ..calibration.roi_config import ROIConfig
from ..domain import Box
from ..perception.config import HSVRange
from ..perception.roi_extractor import extract_roi
from ..perception.metadata_reader import hsv_mask


TEMPLATE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# Bright pale-green/light button fill; must be recalibrated on real client frames.
DEFAULT_HSV_LOWER = (0, 0, 150)
DEFAULT_HSV_UPPER = (179, 90, 255)
DEFAULT_WIDTH_FRACTION = (0.05, 0.9)
DEFAULT_HEIGHT_FRACTION = (0.15, 0.9)
DEFAULT_MINIMUM_FILL_RATIO = 0.5
DEFAULT_NORMALIZED_SIZE = (48, 48)
DEFAULT_MAXIMUM_DISTANCE = 0.25
DEFAULT_MINIMUM_MARGIN = 0.05


@dataclass(frozen=True)
class ButtonConfig:
    hsv_lower: tuple[int, int, int] = DEFAULT_HSV_LOWER
    hsv_upper: tuple[int, int, int] = DEFAULT_HSV_UPPER
    width_fraction: tuple[float, float] = DEFAULT_WIDTH_FRACTION
    height_fraction: tuple[float, float] = DEFAULT_HEIGHT_FRACTION
    minimum_fill_ratio: float = DEFAULT_MINIMUM_FILL_RATIO
    normalized_size: tuple[int, int] = DEFAULT_NORMALIZED_SIZE
    maximum_distance: float = DEFAULT_MAXIMUM_DISTANCE
    minimum_margin: float = DEFAULT_MINIMUM_MARGIN

    @classmethod
    def from_dict(cls, data: dict | None = None) -> "ButtonConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("Button configuration must be a mapping")
        try:
            lower = tuple(int(v) for v in data.get("hsv_lower", DEFAULT_HSV_LOWER))
            upper = tuple(int(v) for v in data.get("hsv_upper", DEFAULT_HSV_UPPER))
            width = tuple(float(v) for v in data.get("width_fraction", DEFAULT_WIDTH_FRACTION))
            height = tuple(float(v) for v in data.get("height_fraction", DEFAULT_HEIGHT_FRACTION))
            size = tuple(int(v) for v in data.get("normalized_size", DEFAULT_NORMALIZED_SIZE))
            value = cls(lower, upper, width, height,
                        float(data.get("minimum_fill_ratio", DEFAULT_MINIMUM_FILL_RATIO)),
                        size,
                        float(data.get("maximum_distance", DEFAULT_MAXIMUM_DISTANCE)),
                        float(data.get("minimum_margin", DEFAULT_MINIMUM_MARGIN)))
        except (TypeError, ValueError) as exc:
            raise ValueError("Malformed button configuration") from exc
        if (len(lower) != 3 or len(upper) != 3 or
                not all(0 <= item <= 179 for item in (lower[0], upper[0])) or
                not all(0 <= item <= 255 for item in (*lower[1:], *upper[1:])) or
                lower[1] > upper[1] or lower[2] > upper[2]):
            raise ValueError("Invalid button HSV range")
        if (len(width) != 2 or len(height) != 2 or
                not 0 < width[0] < width[1] <= 1 or
                not 0 < height[0] < height[1] <= 1):
            raise ValueError("Invalid button size fractions")
        if (not 0 < value.minimum_fill_ratio <= 1 or
                len(size) != 2 or min(size) <= 0 or
                not 0 < value.maximum_distance <= 2 or
                not 0 <= value.minimum_margin < 2):
            raise ValueError("Invalid button classifier configuration")
        return value


def button_descriptor(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Zero-mean normalized grayscale vector for template comparison."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Button image must be a non-empty array")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    vector = resized.astype(np.float32).reshape(-1) / 255
    vector = vector - vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return max(0.0, min(2.0, 1.0 - float(np.dot(a, b)) / (norm_a * norm_b)))


@dataclass(eq=False, frozen=True)
class ButtonCandidate:
    image: np.ndarray
    box: Box
    clipped: bool


def extract_button_candidates(image: np.ndarray, roi: ROIConfig,
                              config: ButtonConfig) -> tuple[ButtonCandidate, ...]:
    """Shared extraction contract for real-frame template export and inference.

    Preserve boundary clipping as evidence: historical full-window imports can
    have different chrome, so their reviewed source ROI must include the icon.
    """
    crop = extract_roi(image, roi, "action_buttons")
    mask = hsv_mask(crop.image, HSVRange(config.hsv_lower, config.hsv_upper))
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    roi_height, roi_width = binary.shape
    candidates = []
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        if not (config.width_fraction[0] * roi_width <= width <=
                config.width_fraction[1] * roi_width):
            continue
        if not (config.height_fraction[0] * roi_height <= height <=
                config.height_fraction[1] * roi_height):
            continue
        if area / (width * height) < config.minimum_fill_ratio:
            continue
        box = crop.map_box_to_screen(Box(x, y, x + width, y + height))
        candidates.append(ButtonCandidate(
            crop.image[y:y + height, x:x + width], box,
            x == 0 or y == 0 or x + width == roi_width or y + height == roi_height))
    return tuple(candidates)


class ActionButtonReader:
    """Detect and label response buttons inside the action_buttons ROI."""

    def __init__(self, config: ButtonConfig, templates: str | Path):
        self.config = config
        self.root = Path(templates)
        self.last_buttons: dict[str, Box] = {}
        self.last_candidates: tuple[ButtonCandidate, ...] = ()
        self.last_occlusion_boxes: tuple[Box, ...] = ()
        self.last_predictions: dict[Box, dict] = {}
        # Instance-level sink: FrameObservationReader calls read() without one.
        self.unknown_sink = None
        self._templates: dict[str, list[np.ndarray]] = defaultdict(list)
        if self.root.exists():
            for path in sorted(self.root.glob("*/*")):
                if path.suffix.lower() not in TEMPLATE_SUFFIXES:
                    continue
                label = path.parent.name
                if not label or any(character in label for character in "/\\"):
                    raise ValueError(f"Invalid button label: {label}")
                image = read_image(path)
                self._templates[label].append(
                    button_descriptor(image, self.config.normalized_size))

    @property
    def known_labels(self) -> tuple[str, ...]:
        return tuple(sorted(self._templates))

    def _prediction(self, descriptor: np.ndarray) -> dict:
        ordered = sorted(
            (min(_cosine_distance(descriptor, template) for template in templates), label)
            for label, templates in self._templates.items())
        best_distance, best_label = ordered[0]
        second_distance = ordered[1][0] if len(ordered) > 1 else 1.0
        margin = second_distance - best_distance
        accepted = (best_distance <= self.config.maximum_distance and
                    (len(ordered) == 1 or margin >= self.config.minimum_margin))
        hint = best_label if best_distance <= self.config.maximum_distance else None
        return {"hint": hint, "distance": best_distance, "margin": margin,
                "accepted": accepted, "ranked_distances": tuple((label, distance) for distance, label in ordered)}

    def _classify(self, descriptor: np.ndarray) -> tuple[str | None, float, bool]:
        result = self._prediction(descriptor)
        return result["hint"], result["distance"], result["accepted"]

    BOUNDARY_PAD = 0.35  # of the ROI height, around a candidate cut by the ROI

    def _reread_clipped(self, image, roi, item):
        """Re-extract a candidate cut by the ROI edge from a padded region.

        Tall calligraphy (the ron button's 和) can reach a few pixels past the
        calibrated row. The padded component replaces the cut one only if it
        lies fully inside the padded region and matches a reviewed template;
        anything else stays clipped and is never clicked.
        """
        name = 'action_buttons'
        x1, y1, x2, y2 = roi.regions[name]
        pad = (y2-y1)*self.BOUNDARY_PAD
        padded = [max(0., x1-pad), max(0., y1-pad), min(1., x2+pad), min(1., y2+pad)]
        try:
            wider = extract_button_candidates(image, replace(roi, regions={**roi.regions, name: padded}),
                                              self.config)
        except (KeyError, ValueError):
            return item
        def overlap(a, b):
            return max(0, min(a.x2, b.x2)-max(a.x1, b.x1))*max(0, min(a.y2, b.y2)-max(a.y1, b.y1))
        best = max(wider, key=lambda other: overlap(other.box, item.box), default=None)
        if best is None or best.clipped or not overlap(best.box, item.box):
            return item
        prediction = self._prediction(button_descriptor(best.image, self.config.normalized_size))
        return best if prediction["accepted"] else item

    def read(self, image: np.ndarray, roi: ROIConfig, *,
             unknown_sink=None) -> dict[str, Box]:
        sink = unknown_sink if unknown_sink is not None else self.unknown_sink
        self.last_buttons = {}
        self.last_candidates = ()
        self.last_occlusion_boxes = ()
        self.last_predictions = {}
        candidates = extract_button_candidates(image, roi, self.config)
        if self._templates and any(item.clipped for item in candidates):
            candidates = tuple(self._reread_clipped(image, roi, item) if item.clipped else item
                               for item in candidates)
        self.last_candidates = candidates
        for item in candidates if self._templates else ():
            prediction = self._prediction(button_descriptor(item.image, self.config.normalized_size))
            prediction["clipped"] = item.clipped
            if item.clipped:
                prediction["accepted"] = False
                # Something cut by the ROI edge that resembles no button
                # (e.g. our own North tiles beside the hand) is table clutter:
                # it neither hides the public row nor becomes a review case.
                prediction["rejection"] = ("ROI_edge_clutter" if prediction["hint"] is None
                                           else "ROI_boundary_clipping")
            self.last_predictions[item.box] = prediction
        ui = [item for item in candidates
              if self.last_predictions.get(item.box, {}).get("rejection") != "ROI_edge_clutter"]
        if ui:
            # Some icons (especially PASS) have disconnected glyph components.
            # Exclude the whole detected UI row, including unknown candidates,
            # rather than feeding the remaining strokes to the tile CNN.
            crop = extract_roi(image, roi, "action_buttons")
            self.last_occlusion_boxes = (Box(
                crop.box.x1, min(item.box.y1 for item in ui),
                crop.box.x2, max(item.box.y2 for item in ui)),)
        buttons: dict[str, tuple[float, Box]] = {}
        for item in ui:
            candidate, screen_box = item.image, item.box
            prediction = self.last_predictions.get(screen_box)
            if prediction is None:
                continue
            label, distance, accepted = prediction["hint"], prediction["distance"], prediction["accepted"]
            if accepted:
                if label not in buttons or distance < buttons[label][0]:
                    buttons[label] = (distance, screen_box)
            elif callable(sink):
                sink(label, candidate, screen_box)
        self.last_buttons = {label: box for label, (_, box) in buttons.items()}
        return self.last_buttons
