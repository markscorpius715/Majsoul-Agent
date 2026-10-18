from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median

import cv2
import numpy as np

from ..calibration.roi_config import ROIConfig
from ..domain import Box
from .config import GlyphStyle, HSVRange, VisionConfig
from .glyph_classifier import GlyphTemplateClassifier
from .roi_extractor import ROICrop, extract_roi, rotate_roi
from .schemas import (GameMetadataReading, GlyphPrediction, NumericReading,
                      RoundReading, SeatReading, SegmentedGlyph)


WIND_LABELS = {
    "wind_east": "1z",
    "wind_south": "2z",
    "wind_west": "3z",
    "wind_north": "4z",
}
DIGIT_LABELS = frozenset("0123456789")


def seat_consistency_issues(seats: dict[str, SeatReading],
                            player_count: int) -> tuple[str, ...]:
    """Validate every fact already known, even when another seat is unknown."""
    issues = []
    winds = [seat.wind for seat in seats.values() if seat.wind is not None]
    if len(set(winds)) != len(winds):
        issues.append("Active layout contains duplicate known seat winds")
    if len(winds) == player_count:
        expected = set(("1z", "2z", "3z", "4z")[:player_count])
        if set(winds) != expected:
            issues.append("Seat winds do not match the configured player count")
    known_dealers = [seat.dealer for seat in seats.values()
                     if seat.dealer is not None]
    dealer_count = sum(value is True for value in known_dealers)
    if dealer_count > 1:
        issues.append("Active layout contains multiple known dealers")
    if len(known_dealers) == player_count and dealer_count != 1:
        issues.append("Active layout must contain exactly one dealer")
    return tuple(issues)


def hsv_mask(image: np.ndarray, color: HSVRange) -> np.ndarray:
    """Return a binary HSV mask, including hue ranges that wrap across red."""
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or
            image.shape[2] != 3 or image.size == 0):
        raise ValueError("HSV masking requires a non-empty BGR image")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.asarray(color.lower, dtype=np.uint8)
    upper = np.asarray(color.upper, dtype=np.uint8)
    if color.lower[0] <= color.upper[0]:
        return cv2.inRange(hsv, lower, upper)
    high_red = cv2.inRange(
        hsv, lower, np.asarray((179, color.upper[1], color.upper[2]), dtype=np.uint8))
    low_red = cv2.inRange(
        hsv, np.asarray((0, color.lower[1], color.lower[2]), dtype=np.uint8), upper)
    return cv2.bitwise_or(high_red, low_red)


def _tight_box(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    piece = mask[y1:y2, x1:x2]
    points = cv2.findNonZero(piece)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return x1 + x, y1 + y, x1 + x + width, y1 + y + height


def _split_wide_box(mask: np.ndarray, box, typical_width: float):
    x, y, width, height = box
    parts = max(2, min(6, round(width / max(1.0, typical_width))))
    projection = np.count_nonzero(mask[y:y + height, x:x + width], axis=0)
    boundaries = [0]
    for index in range(1, parts):
        target = width * index / parts
        radius = max(1, round(typical_width * 0.30))
        start = max(boundaries[-1] + 1, round(target) - radius)
        stop = min(width - (parts - index), round(target) + radius + 1)
        if stop <= start:
            boundary = round(target)
        else:
            boundary = start + int(np.argmin(projection[start:stop]))
        boundaries.append(boundary)
    boundaries.append(width)
    result = []
    for left, right in zip(boundaries, boundaries[1:]):
        tightened = _tight_box(mask, x + left, y, x + right, y + height)
        if tightened is not None:
            result.append(tightened)
    return result


class GlyphSegmenter:
    """Segment fixed-font glyphs from a calibrated, optionally rotated ROI."""

    def __init__(self, config: VisionConfig):
        self.config = config

    def color_mask(self, crop: ROICrop, style_name: str) -> np.ndarray:
        try:
            style = self.config.glyph_styles[style_name]
        except KeyError as exc:
            raise ValueError(f"Unknown glyph style: {style_name}") from exc
        return hsv_mask(crop.image, style.color)

    def segment(self, crop: ROICrop, style_name: str, *,
                split_wide: bool = False) -> tuple[SegmentedGlyph, ...]:
        try:
            style = self.config.glyph_styles[style_name]
        except KeyError as exc:
            raise ValueError(f"Unknown glyph style: {style_name}") from exc
        return self.segment_mask(crop, self.color_mask(crop, style_name), style,
                                 split_wide=split_wide)

    def segment_mask(self, crop: ROICrop, mask: np.ndarray, style: GlyphStyle, *,
                     split_wide: bool = False) -> tuple[SegmentedGlyph, ...]:
        if mask.ndim != 2 or mask.shape != crop.image.shape[:2]:
            raise ValueError("Glyph mask dimensions must match the ROI")
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        minimum_height = math.ceil(binary.shape[0] * style.minimum_height_fraction)
        components = []
        for label in range(1, count):
            x, y, width, height, area = map(int, stats[label])
            if area >= style.minimum_area and height >= minimum_height:
                components.append((x, y, width, height, area, label))
        components.sort(key=lambda item: (item[0], item[1]))

        boxes = [(x, y, width, height) for x, y, width, height, _, _ in components]
        if split_wide and boxes:
            widths = [width for _, _, width, _ in boxes]
            typical_width = float(median(widths))
            if len(widths) == 1:
                typical_width = min(typical_width, boxes[0][3] * 0.58)
            expanded = []
            for box in boxes:
                if box[2] >= typical_width * 1.65:
                    expanded.extend(_split_wide_box(binary, box, typical_width))
                else:
                    x, y, width, height = box
                    expanded.append((x, y, x + width, y + height))
            pixel_boxes = expanded
        else:
            pixel_boxes = [(x, y, x + width, y + height)
                           for x, y, width, height in boxes]

        glyphs = []
        for x1, y1, x2, y2 in sorted(pixel_boxes):
            local_mask = binary[y1:y2, x1:x2].copy()
            screen_box = crop.map_box_to_screen(Box(x1, y1, x2, y2))
            glyphs.append(SegmentedGlyph(screen_box, local_mask))
        return tuple(glyphs)

    def segment_merged(self, crop: ROICrop, mask: np.ndarray,
                       style: GlyphStyle, *, exclude_border=False) -> SegmentedGlyph | None:
        """Merge valid components when one ROI is known to contain one glyph.

        Seat-wind ideographs can contain disconnected strokes (notably 北).
        Numeric fields continue to use ``segment_mask`` because those ROIs
        intentionally contain multiple characters.
        """
        if mask.ndim != 2 or mask.shape != crop.image.shape[:2]:
            raise ValueError("Glyph mask dimensions must match the ROI")
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        minimum_height = math.ceil(binary.shape[0] * style.minimum_height_fraction)
        accepted = []
        for label in range(1, count):
            x, y, width, height, area = map(int, stats[label])
            if exclude_border and (x == 0 or y == 0 or
                                   x + width == binary.shape[1] or
                                   y + height == binary.shape[0]):
                continue
            if area >= style.minimum_area and height >= minimum_height:
                accepted.append(label)
        if not accepted:
            return None
        merged = np.where(np.isin(labels, accepted), 255, 0).astype(np.uint8)
        points = cv2.findNonZero(merged)
        x, y, width, height = cv2.boundingRect(points)
        local_mask = merged[y:y + height, x:x + width].copy()
        return SegmentedGlyph(
            crop.map_box_to_screen(Box(x, y, x + width, y + height)),
            local_mask)


@dataclass(frozen=True)
class _NumericSpec:
    style: str
    prefix: str | None = None
    split_wide: bool = False


class GameMetadataReader:
    """Read numeric table metadata for an already locked match-level layout."""

    def __init__(self, config: VisionConfig, classifier: GlyphTemplateClassifier, *,
                 read_scores: bool = True, numeric_ocr=None):
        self.config = config
        self.classifier = classifier
        self.segmenter = GlyphSegmenter(config)
        # Live pipelines may disable score OCR entirely: scores then stay
        # unavailable (None) instead of blocking layout and state tracking.
        self.read_scores = read_scores
        self.numeric_ocr = numeric_ocr

    def _predictions(self, glyphs, allowed_labels=None) -> tuple[GlyphPrediction, ...]:
        return tuple(self.classifier.classify(
            glyph.mask, glyph.box, allowed_labels=allowed_labels) for glyph in glyphs)

    @staticmethod
    def _integer(labels: list[str | None]):
        if not labels:
            return None
        sign = 1
        if labels[0] == "minus":
            sign = -1
            labels = labels[1:]
        if not labels or any(label not in DIGIT_LABELS for label in labels):
            return None
        return sign * int("".join(labels))

    def _read_numeric(self, crop: ROICrop, spec: _NumericSpec) -> NumericReading:
        glyphs = self.segmenter.segment(crop, spec.style, split_wide=spec.split_wide)
        predictions = []
        for index, glyph in enumerate(glyphs):
            allowed = ({spec.prefix} if index == 0 and spec.prefix is not None
                       else DIGIT_LABELS | {"minus"})
            predictions.append(self.classifier.classify(
                glyph.mask, glyph.box, allowed_labels=allowed))
        predictions = tuple(predictions)
        issues = []
        offset = 0
        if spec.prefix is not None:
            offset = 1
            if not predictions:
                issues.append("Missing prefix and numeric glyphs")
            elif predictions[0].label != spec.prefix:
                issues.append(f"Unrecognized {spec.prefix} prefix")
        labels = [prediction.label for prediction in predictions[offset:]]
        if self.numeric_ocr is not None and crop.name in {'remaining_tiles', 'honba', 'riichi_sticks'}:
            digits = glyphs[offset:]
            if not digits or crop.rotation_degrees:
                return NumericReading(crop.name, None, predictions,
                                      tuple(issues + ['Numeric OCR glyph geometry unavailable']))
            x1 = int(min(g.box.x1 for g in digits)-crop.box.x1)
            y1 = int(min(g.box.y1 for g in digits)-crop.box.y1)
            x2 = int(max(g.box.x2 for g in digits)-crop.box.x1)
            y2 = int(max(g.box.y2 for g in digits)-crop.box.y1)
            if x1 <= 0 or y1 <= 0 or x2 >= crop.image.shape[1] or y2 >= crop.image.shape[0]:
                return NumericReading(crop.name, None, predictions,
                                      tuple(issues + ['Numeric OCR glyphs touch ROI boundary']))
            patch = crop.image[y1:y2, x1:x2]
            read_field = getattr(self.numeric_ocr, 'read_for_field', None)
            text, confidence, error = (read_field(patch, crop.name) if read_field is not None
                                       else self.numeric_ocr.read(patch))
            if error or text is None or len(text) != len(digits):
                return NumericReading(crop.name, None, predictions,
                                      tuple(issues + [error or 'Numeric OCR segmentation mismatch']))
            labels = list(text)
            predictions = (*predictions[:offset], *(GlyphPrediction(
                label, confidence, None, None, glyph.box) for label, glyph in zip(labels, digits)))
        value = self._integer(labels)
        if value is None:
            issues.append("Numeric glyph sequence is incomplete or invalid")
        return NumericReading(crop.name, value, predictions, tuple(issues))

    def _read_round(self, crop: ROICrop) -> RoundReading:
        glyphs = self.segmenter.segment(crop, "center")
        allowed = (WIND_LABELS, DIGIT_LABELS, {"suffix_round"})
        predictions = tuple(self.classifier.classify(
            glyph.mask, glyph.box,
            allowed_labels=allowed[index] if index < len(allowed) else None)
                            for index, glyph in enumerate(glyphs))
        issues = []
        round_wind = None
        hand_number = None
        if len(glyphs) != 3:
            issues.append(f"Expected 3 round glyphs, found {len(glyphs)}")
        else:
            round_wind = WIND_LABELS.get(predictions[0].label)
            if round_wind is None:
                issues.append("Round wind glyph is unknown")
            label = predictions[1].label
            if label in set("1234"):
                hand_number = int(label)
            else:
                issues.append("Round hand number is unknown")
            if predictions[2].label != "suffix_round":
                issues.append("Round suffix glyph is unknown")
        return RoundReading(round_wind, hand_number, predictions, tuple(issues))

    def _dealer_glyph_mask(self, crop: ROICrop, red_mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(red_mask.shape, dtype=np.uint8)
        tile = max(contours, key=cv2.contourArea)
        interior = np.zeros(red_mask.shape, dtype=np.uint8)
        cv2.drawContours(interior, [tile], -1, 255, cv2.FILLED)
        radius = max(1, round(min(red_mask.shape) * 0.02))
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        interior = cv2.erode(interior, kernel, iterations=1)
        value = cv2.cvtColor(crop.image, cv2.COLOR_BGR2HSV)[:, :, 2]
        dark = cv2.inRange(value, 0, self.config.dealer_glyph_value_max)
        return cv2.bitwise_and(dark, interior)

    def _read_seat(self, crop: ROICrop, slot: str) -> SeatReading:
        red_mask = hsv_mask(crop.image, self.config.dealer_red)
        red_fill_ratio = cv2.countNonZero(red_mask) / red_mask.size
        issues = []
        if red_fill_ratio >= self.config.dealer_minimum_fill_ratio:
            dealer = True
            mask = self._dealer_glyph_mask(crop, red_mask)
        elif red_fill_ratio <= self.config.dealer_minimum_fill_ratio * 0.25:
            dealer = False
            mask = self.segmenter.color_mask(crop, "seat")
        else:
            dealer = None
            mask = self.segmenter.color_mask(crop, "seat")
            issues.append("Dealer color is ambiguous")
        glyph = self.segmenter.segment_merged(
            crop, mask, self.config.glyph_styles["seat"], exclude_border=True)
        prediction = (self.classifier.classify(
            glyph.mask, glyph.box, allowed_labels=WIND_LABELS)
                      if glyph is not None else None)
        if glyph is None:
            issues.append("Expected 1 seat glyph, found 0")
        wind = WIND_LABELS.get(prediction.label) if prediction is not None else None
        if wind is None:
            issues.append("Seat wind glyph is unknown")
        return SeatReading(slot, wind, dealer, prediction, red_fill_ratio, tuple(issues))

    def read(self, image: np.ndarray, roi: ROIConfig, active_slots) -> GameMetadataReading:
        slots = tuple(roi.layout_for_active_slots(active_slots))
        round_reading = self._read_round(extract_roi(image, roi, "round_info"))
        remaining = self._read_numeric(
            extract_roi(image, roi, "remaining_tiles"),
            _NumericSpec("center", "prefix_remaining"))
        honba = self._read_numeric(
            extract_roi(image, roi, "honba"),
            _NumericSpec("counter", "prefix_multiply"))
        riichi_sticks = self._read_numeric(
            extract_roi(image, roi, "riichi_sticks"),
            _NumericSpec("counter", "prefix_multiply"))
        scores = {}
        seats = {}
        for slot in slots:
            degrees = self.config.rotation_for_slot(slot)
            if self.read_scores:
                score_crop = rotate_roi(extract_roi(image, roi, f"score_{slot}"), degrees)
                scores[slot] = self._read_numeric(
                    score_crop, _NumericSpec("score", split_wide=True))
            else:
                scores[slot] = NumericReading(f"score_{slot}", None, ())
            seat_crop = rotate_roi(extract_roi(image, roi, f"seat_{slot}"), degrees)
            seats[slot] = self._read_seat(seat_crop, slot)

        issues = list(seat_consistency_issues(seats, roi.player_count))
        return GameMetadataReading(
            slots, round_reading, remaining, honba, riichi_sticks,
            scores, seats, tuple(issues), scores_available=self.read_scores)
