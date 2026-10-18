from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ComponentLimits:
    width_fraction: tuple[float, float]
    height_fraction: tuple[float, float]
    max_tiles: int

    @classmethod
    def from_mapping(cls, data):
        try:
            width = tuple(float(v) for v in data["width_fraction"])
            height = tuple(float(v) for v in data["height_fraction"])
            value = cls(width, height, int(data["max_tiles"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed component limits") from exc
        if (len(width) != 2 or len(height) != 2 or
                not 0 < width[0] < width[1] <= 1 or
                not 0 < height[0] < height[1] <= 1 or value.max_tiles <= 0):
            raise ValueError("Invalid component limits")
        return value


@dataclass(frozen=True)
class HSVRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    @classmethod
    def from_mapping(cls, data):
        try:
            lower = tuple(int(value) for value in data["lower"])
            upper = tuple(int(value) for value in data["upper"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed HSV range") from exc
        value = cls(lower, upper)
        if (len(lower) != 3 or len(upper) != 3 or
                not all(0 <= item <= 179 for item in (lower[0], upper[0])) or
                not all(0 <= item <= 255 for item in (*lower[1:], *upper[1:])) or
                lower[1] > upper[1] or lower[2] > upper[2]):
            raise ValueError("Invalid HSV range")
        return value


@dataclass(frozen=True)
class GlyphStyle:
    color: HSVRange
    minimum_height_fraction: float
    minimum_area: int

    @classmethod
    def from_mapping(cls, data):
        try:
            value = cls(
                HSVRange.from_mapping(data),
                float(data["minimum_height_fraction"]),
                int(data["minimum_area"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed glyph style") from exc
        if not 0 < value.minimum_height_fraction <= 1 or value.minimum_area <= 0:
            raise ValueError("Invalid glyph style")
        return value


@dataclass(frozen=True)
class VisionConfig:
    saturation_max: int
    value_min: int
    minimum_fill_ratio: float
    upright_minimum_aspect_ratio: float
    hand: ComponentLimits
    river: ComponentLimits
    drawn_gap_ratio: float
    row_tolerance_ratio: float
    maximum_row_gap_ratio: float
    horizontal_aspect_ratio: float
    slot_rotation_degrees: dict[str, int]
    dora: ComponentLimits
    kita: dict[str, ComponentLimits]
    meld: dict[str, ComponentLimits]
    meld_group_gap_ratio: float
    normalized_size: tuple[int, int]
    maximum_distance: float
    minimum_margin: float
    glyph_normalized_size: tuple[int, int]
    glyph_maximum_distance: float
    glyph_minimum_margin: float
    glyph_styles: dict[str, GlyphStyle]
    dealer_red: HSVRange
    dealer_minimum_fill_ratio: float
    dealer_glyph_value_max: int
    schema_version: int = 1
    back_hsv: HSVRange | None = None
    back_minimum_fill_ratio: float = 0.6
    river_outline_opening_fraction: float = 0.0

    @classmethod
    def load(cls, path: str | Path) -> "VisionConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Unsupported or malformed vision configuration")
        try:
            face = data["tile_face"]
            hand_data = data["hand"]
            river_data = data["river"]
            classifier = data["classifier"]
            metadata = data["metadata"]
            value = cls(
                saturation_max=int(face["saturation_max"]),
                value_min=int(face["value_min"]),
                minimum_fill_ratio=float(face["minimum_fill_ratio"]),
                upright_minimum_aspect_ratio=float(face["upright_minimum_aspect_ratio"]),
                hand=ComponentLimits.from_mapping(hand_data),
                river=ComponentLimits.from_mapping(river_data),
                drawn_gap_ratio=float(hand_data["drawn_gap_ratio"]),
                row_tolerance_ratio=float(river_data["row_tolerance_ratio"]),
                maximum_row_gap_ratio=float(river_data["maximum_row_gap_ratio"]),
                horizontal_aspect_ratio=float(river_data["horizontal_aspect_ratio"]),
                slot_rotation_degrees={str(key): int(value) for key, value in
                                       data["slot_rotation_degrees"].items()},
                dora=ComponentLimits.from_mapping(data["dora"]),
                kita={str(key): ComponentLimits.from_mapping(value)
                      for key, value in data["kita"].items()},
                meld={str(key): ComponentLimits.from_mapping(value)
                      for key, value in data["meld"].items()
                      if key != "group_gap_ratio"},
                meld_group_gap_ratio=float(data["meld"]["group_gap_ratio"]),
                normalized_size=tuple(int(v) for v in classifier["normalized_size"]),
                maximum_distance=float(classifier["maximum_distance"]),
                minimum_margin=float(classifier["minimum_margin"]),
                glyph_normalized_size=tuple(int(v) for v in metadata["normalized_size"]),
                glyph_maximum_distance=float(metadata["maximum_distance"]),
                glyph_minimum_margin=float(metadata["minimum_margin"]),
                glyph_styles={str(key): GlyphStyle.from_mapping(item)
                              for key, item in metadata["styles"].items()},
                dealer_red=HSVRange.from_mapping(metadata["dealer_red"]),
                dealer_minimum_fill_ratio=float(metadata["dealer_minimum_fill_ratio"]),
                dealer_glyph_value_max=int(metadata["dealer_glyph_value_max"]),
                back_hsv=(HSVRange.from_mapping(data['tile_back'])
                          if data.get('tile_back') else None),
                back_minimum_fill_ratio=float((data.get('tile_back') or {}).get('minimum_fill_ratio', .6)),
                river_outline_opening_fraction=float(river_data.get('outline_opening_fraction', 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed vision configuration") from exc
        if (not 0 <= value.river_outline_opening_fraction <= .5 or
                not 0 < value.back_minimum_fill_ratio <= 1 or
                not 0 <= value.saturation_max <= 255 or not 0 <= value.value_min <= 255 or
                not 0 < value.minimum_fill_ratio <= 1 or
                not 0 < value.upright_minimum_aspect_ratio <= 2 or
                value.drawn_gap_ratio <= 1 or
                not 0 < value.row_tolerance_ratio <= 1 or
                value.maximum_row_gap_ratio <= 1 or value.horizontal_aspect_ratio <= 1 or
                len(value.normalized_size) != 2 or min(value.normalized_size) <= 0 or
                not 0 < value.maximum_distance <= 1 or not 0 <= value.minimum_margin < 1):
            raise ValueError("Invalid vision configuration")
        if (len(value.glyph_normalized_size) != 2 or min(value.glyph_normalized_size) <= 0 or
                not 0 < value.glyph_maximum_distance <= 1 or
                not 0 <= value.glyph_minimum_margin < 1 or
                not 0 < value.dealer_minimum_fill_ratio <= 1 or
                not 0 <= value.dealer_glyph_value_max <= 255 or
                set(value.glyph_styles) != {"score", "center", "counter", "seat"}):
            raise ValueError("Invalid metadata vision configuration")
        if (set(value.slot_rotation_degrees) != {"bottom", "top", "left", "right"} or
                any(rotation not in (-90, 0, 90, 180)
                    for rotation in value.slot_rotation_degrees.values())):
            raise ValueError("Invalid slot rotations")
        if set(value.kita) != {"bottom", "top", "left", "right"}:
            raise ValueError("Invalid kita component limits")
        if (set(value.meld) != {"bottom", "top", "left", "right"} or
                value.meld_group_gap_ratio <= 1):
            raise ValueError("Invalid meld configuration")
        return value

    def rotation_for_slot(self, slot: str) -> int:
        try:
            return self.slot_rotation_degrees[slot]
        except KeyError as exc:
            raise ValueError(f"Unknown screen slot: {slot}") from exc

    def kita_for_slot(self, slot: str) -> ComponentLimits:
        try:
            return self.kita[slot]
        except KeyError as exc:
            raise ValueError(f"Unknown screen slot: {slot}") from exc

    def meld_for_slot(self, slot: str) -> ComponentLimits:
        try:
            return self.meld[slot]
        except KeyError as exc:
            raise ValueError(f"Unknown screen slot: {slot}") from exc
