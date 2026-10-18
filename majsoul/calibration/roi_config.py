from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
import re

import yaml


GLOBAL_REGIONS = [
    "round_info",
    "remaining_tiles",
    "honba",
    "riichi_sticks",
    "dora_indicators",
    "self_hand",
    "action_buttons",
    "turn_indicator",
]
SLOT_REGION_KINDS = ("score", "seat", "river", "melds")
SANMA_SCREEN_SLOTS = ["bottom", "top", "left", "right"]
SANMA_LAYOUT_VARIANTS = [
    ["bottom", "top", "left"],
    ["bottom", "top", "right"],
    ["bottom", "left", "right"],
]

PROJECTION_NAMES = (
    "stretch",
    "fit_width_top",
    "fit_width_center",
    "fit_width_bottom",
    "fit_height_left",
    "fit_height_center",
    "fit_height_right",
)


@dataclass(frozen=True)
class TargetSizeAssessment:
    reference_size: tuple[int, int]
    target_size: tuple[int, int]
    aspect_drift: float
    status: str
    auto_tolerance: float

    @property
    def requires_trials(self) -> bool:
        return self.status == "review_required"

    def to_dict(self) -> dict:
        return {
            "reference_size": list(self.reference_size),
            "target_size": list(self.target_size),
            "aspect_drift": self.aspect_drift,
            "status": self.status,
            "auto_tolerance": self.auto_tolerance,
            "requires_projection_trials": self.requires_trials,
        }


@dataclass(frozen=True)
class ROIProjection:
    name: str
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    boxes: dict[str, tuple[int, int, int, int]]
    clipped_regions: tuple[str, ...]

    @property
    def anisotropy(self) -> float:
        return abs(self.scale_x / self.scale_y - 1.0)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scale": [self.scale_x, self.scale_y],
            "offset": [self.offset_x, self.offset_y],
            "anisotropy": self.anisotropy,
            "clipped_regions": list(self.clipped_regions),
        }


def default_layout(player_count: int) -> tuple[list[str], list[list[str]], str]:
    """Return screen slots, legal occupied-slot layouts, and the self slot."""
    if player_count == 3:
        return (list(SANMA_SCREEN_SLOTS),
                [list(variant) for variant in SANMA_LAYOUT_VARIANTS], "bottom")
    if player_count == 4:
        slots = list(SANMA_SCREEN_SLOTS)
        return slots, [list(slots)], "bottom"
    raise ValueError("player_count must be 3 or 4")


def required_regions(screen_slots: list[str], kita: bool) -> list[str]:
    """Return every ROI that must be calibrated across all layout variants."""
    if (not isinstance(screen_slots, list) or not screen_slots or
            any(not isinstance(slot, str) or not slot.strip() for slot in screen_slots) or
            len(set(screen_slots)) != len(screen_slots)):
        raise ValueError("screen_slots must contain unique non-empty labels")
    if type(kita) is not bool:
        raise ValueError("kita must be boolean")
    # The drawn tile is derived inside self_hand from spacing and time. Dealer
    # state is the red/normal appearance of the same seat marker.
    names = list(GLOBAL_REGIONS)
    for slot in screen_slots:
        names.extend(f"{kind}_{slot}" for kind in SLOT_REGION_KINDS)
        if kita:
            names.append(f"kita_{slot}")
    return names


def _migrate_v1(data: dict) -> dict:
    """Migrate numeric player ROIs to physical screen slots without guessing."""
    player_count = data.get("player_count", 3)
    screen_slots, layout_variants, self_slot = default_layout(player_count)
    migrated = dict.fromkeys(required_regions(screen_slots, bool(data.get("kita", False))))
    legacy_slots = {"0": "bottom", "1": "top", "2": "left", "3": "right"}
    old_regions = data.get("regions", {})
    if not isinstance(old_regions, dict):
        raise ValueError("Malformed ROI configuration")
    for name, rect in old_regions.items():
        if name == "drawn_tile" or re.fullmatch(r"dealer_\d+", str(name)):
            continue
        match = re.fullmatch(r"(score|seat|river|melds|kita)_(\d+)", str(name))
        if match and match.group(2) in legacy_slots:
            migrated[f"{match.group(1)}_{legacy_slots[match.group(2)]}"] = rect
        else:
            migrated[name] = rect
    return {
        "reference_size": data.get("reference_size"),
        "player_count": player_count,
        "screen_slots": screen_slots,
        "layout_variants": layout_variants,
        "self_slot": self_slot,
        "kita": data.get("kita", False),
        "regions": migrated,
        "schema_version": 2,
    }


@dataclass
class ROIConfig:
    reference_size: list[int]
    player_count: int
    screen_slots: list[str]
    layout_variants: list[list[str]]
    self_slot: str
    kita: bool
    regions: dict[str, list[float] | None]
    max_auto_aspect_drift: float = 0.03
    schema_version: int = 2
    region_polygons: dict[str, list[list[float]]] = field(default_factory=dict)

    def validate(self, complete=False):
        if self.schema_version != 2:
            raise ValueError("Unsupported ROI schema version")
        if (not isinstance(self.reference_size, list) or len(self.reference_size) != 2 or
                any(type(v) is not int or v <= 0 for v in self.reference_size)):
            raise ValueError("reference_size must be [positive width, positive height]")
        if self.player_count not in (3, 4):
            raise ValueError("player_count must be 3 or 4")
        if (type(self.max_auto_aspect_drift) not in (float, int) or
                not math.isfinite(self.max_auto_aspect_drift) or
                not 0 < self.max_auto_aspect_drift <= 0.25):
            raise ValueError("max_auto_aspect_drift must be in (0, 0.25]")
        names = required_regions(self.screen_slots, self.kita)
        if self.self_slot not in self.screen_slots:
            raise ValueError("self_slot must be one of screen_slots")
        if not isinstance(self.layout_variants, list) or not self.layout_variants:
            raise ValueError("layout_variants must contain at least one layout")
        seen = set()
        covered = set()
        for variant in self.layout_variants:
            if (not isinstance(variant, list) or len(variant) != self.player_count or
                    len(set(variant)) != self.player_count or
                    any(slot not in self.screen_slots for slot in variant) or
                    self.self_slot not in variant):
                raise ValueError("Each layout variant must contain self and exactly player_count unique slots")
            key = frozenset(variant)
            if key in seen:
                raise ValueError("layout_variants must be unique")
            seen.add(key)
            covered.update(variant)
        if covered != set(self.screen_slots):
            raise ValueError("layout_variants must cover every screen slot")
        if not isinstance(self.regions, dict):
            raise ValueError("regions must be a mapping")
        for name, rect in self.regions.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError("ROI names must use letters, numbers and underscores")
            if rect is None:
                continue
            if (not isinstance(rect, (list, tuple)) or len(rect) != 4 or
                    any(type(v) not in (float, int) or not math.isfinite(v) for v in rect)):
                raise ValueError(f"{name}: expected four finite normalized coordinates")
            x1, y1, x2, y2 = rect
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"{name}: invalid normalized rectangle")
        if not isinstance(self.region_polygons, dict):
            raise ValueError('region_polygons must be a mapping')
        for name, points in self.region_polygons.items():
            rect = self.regions.get(name)
            if rect is None or not isinstance(points, list) or len(points) < 3:
                raise ValueError(f'{name}: polygon requires a calibrated ROI and at least three vertices')
            for point in points:
                if (not isinstance(point, (list, tuple)) or len(point) != 2
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
                        or not rect[0] <= point[0] <= rect[2]
                        or not rect[1] <= point[1] <= rect[3]):
                    raise ValueError(f'{name}: invalid polygon vertex')
            area = sum(a[0]*b[1]-b[0]*a[1]
                       for a, b in zip(points, points[1:]+points[:1]))
            if abs(area) <= 1e-12:
                raise ValueError(f'{name}: polygon has zero area')
        missing = [name for name in names if self.regions.get(name) is None]
        if complete and missing:
            raise ValueError("Uncalibrated ROIs: " + ", ".join(missing))
        return missing

    def layout_for_active_slots(self, active_slots) -> list[str]:
        """Match observed occupied screen slots to one configured layout."""
        active = list(active_slots)
        if (len(active) != self.player_count or len(set(active)) != len(active) or
                any(slot not in self.screen_slots for slot in active)):
            raise ValueError("Active slots do not form a valid player layout")
        key = frozenset(active)
        for variant in self.layout_variants:
            if frozenset(variant) == key:
                return list(variant)
        raise ValueError("Active slots do not match a configured layout variant")

    def active_region_names(self, active_slots) -> list[str]:
        """Return global ROIs plus slot ROIs valid for the detected layout."""
        active = set(self.layout_for_active_slots(active_slots))
        names = []
        for name in self.regions:
            slot = next((slot for slot in self.screen_slots if name.endswith(f"_{slot}")), None)
            if slot is None or slot in active:
                names.append(name)
        return names

    def for_game_mode(self, player_count: int, *, kita: bool | None = None) -> "ROIConfig":
        """Reuse calibrated screen geometry with a different player-count contract.

        Geometry belongs to the client layout, while occupied slots and kita are
        game-mode semantics.  Keeping this conversion explicit prevents four-player
        footage from being interpreted with three-player rules.
        """
        slots, variants, self_slot = default_layout(player_count)
        if set(slots) != set(self.screen_slots):
            raise ValueError("Game-mode override requires the same physical screen slots")
        if kita is None:
            kita = self.kita if player_count == self.player_count else False
        if type(kita) is not bool:
            raise ValueError("kita override must be boolean")
        converted = replace(
            self, player_count=player_count, screen_slots=slots,
            layout_variants=variants, self_slot=self_slot, kita=kita)
        converted.validate()
        return converted

    def save(self, path):
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Malformed ROI configuration")
        try:
            if data.get("schema_version", 1) == 1:
                data = _migrate_v1(data)
            config = cls(**data)
            config.validate()
        except (TypeError, AttributeError, KeyError) as exc:
            raise ValueError("Malformed ROI configuration") from exc
        return config

    def assess_target_size(self, size) -> TargetSizeAssessment:
        """Classify whether normalized x/y scaling is safe for a target frame.

        Resolution changes alone are harmless because ROI coordinates are stored as
        independent x/y fractions.  A material aspect-ratio change is different: it
        can mean that the recording included or removed window chrome.  Such frames
        require the explicit multi-projection trial path instead of silently choosing
        one crop convention.
        """
        self.validate()
        if (not isinstance(size, (list, tuple)) or len(size) != 2 or
                any(type(value) is not int or value <= 0 for value in size)):
            raise ValueError("Target size must be [positive integer width, height]")
        width, height = size
        rw, rh = self.reference_size
        reference_aspect = rw / rh
        target_aspect = width / height
        drift = abs(target_aspect / reference_aspect - 1.0)
        if abs(width - height * reference_aspect) <= 1:
            status = "exact_aspect"
        elif drift <= self.max_auto_aspect_drift:
            status = "normalized_xy"
        else:
            status = "review_required"
        return TargetSizeAssessment(
            (rw, rh), (width, height), drift, status,
            float(self.max_auto_aspect_drift))

    def _projection_transform(self, size, name: str):
        width, height = size
        rw, rh = self.reference_size
        if name == "stretch":
            return width / rw, height / rh, 0.0, 0.0
        if name.startswith("fit_width_"):
            scale = width / rw
            spare = height - rh * scale
            alignment = name.removeprefix("fit_width_")
            offsets = {"top": 0.0, "center": spare / 2, "bottom": spare}
            if alignment not in offsets:
                raise ValueError(f"Unknown ROI projection: {name}")
            return scale, scale, 0.0, offsets[alignment]
        if name.startswith("fit_height_"):
            scale = height / rh
            spare = width - rw * scale
            alignment = name.removeprefix("fit_height_")
            offsets = {"left": 0.0, "center": spare / 2, "right": spare}
            if alignment not in offsets:
                raise ValueError(f"Unknown ROI projection: {name}")
            return scale, scale, offsets[alignment], 0.0
        raise ValueError(f"Unknown ROI projection: {name}")

    def projection(self, size, name: str = "auto") -> ROIProjection:
        assessment = self.assess_target_size(size)
        if name == "auto":
            if assessment.requires_trials:
                percent = assessment.aspect_drift * 100
                raise ValueError(
                    f"Target aspect ratio drift is {percent:.2f}% (automatic limit "
                    f"{self.max_auto_aspect_drift * 100:.2f}%); run calibration "
                    "debug with --projection all and review the multiple ROI crops")
            name = "stretch"
        if name not in PROJECTION_NAMES:
            raise ValueError(
                "projection must be auto or one of: " + ", ".join(PROJECTION_NAMES))

        width, height = assessment.target_size
        rw, rh = assessment.reference_size
        scale_x, scale_y, offset_x, offset_y = self._projection_transform(
            assessment.target_size, name)
        boxes = {}
        clipped = []
        for region, rect in self.regions.items():
            if rect is None:
                continue
            raw = (
                rect[0] * rw * scale_x + offset_x,
                rect[1] * rh * scale_y + offset_y,
                rect[2] * rw * scale_x + offset_x,
                rect[3] * rh * scale_y + offset_y,
            )
            limited = (
                max(0, min(width, math.floor(raw[0]))),
                max(0, min(height, math.floor(raw[1]))),
                max(0, min(width, math.ceil(raw[2]))),
                max(0, min(height, math.ceil(raw[3]))),
            )
            if limited != tuple((math.floor(raw[0]), math.floor(raw[1]),
                                 math.ceil(raw[2]), math.ceil(raw[3]))):
                clipped.append(region)
            if limited[0] < limited[2] and limited[1] < limited[3]:
                boxes[region] = limited
            elif region not in clipped:
                clipped.append(region)
        return ROIProjection(name, scale_x, scale_y, offset_x, offset_y,
                             boxes, tuple(clipped))

    def projection_candidates(self, size) -> tuple[ROIProjection, ...]:
        """Return distinct crop hypotheses for explicit geometry review."""
        candidates = []
        fingerprints = set()
        for name in PROJECTION_NAMES:
            candidate = self.projection(size, name)
            fingerprint = tuple(sorted(candidate.boxes.items()))
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            candidates.append(candidate)
        return tuple(candidates)

    def pixel_boxes(self, size, *, projection: str = "auto"):
        return self.projection(size, projection).boxes
