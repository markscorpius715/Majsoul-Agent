"""Read the client clock and keep one decreasing budget across substeps."""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from ..domain import Box
from ..perception.config import HSVRange
from ..perception.glyph_classifier import GlyphTemplateClassifier
from ..perception.metadata_reader import hsv_mask
from ..perception.roi_extractor import extract_roi


@dataclass(frozen=True)
class CountdownConfig:
    templates: Path
    roi: str
    white: HSVRange
    gold: HSVRange
    minimum_height_fraction: float
    minimum_area_fraction: float
    baseline_tolerance: float
    base_gap_ratio: float
    normalized_size: tuple[int, int]
    maximum_distance: float
    minimum_margin: float
    confirmations: int
    quantization_margin_seconds: float
    maximum_sample_gap_seconds: float

    @classmethod
    def load(cls, path, root):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Unsupported countdown configuration")
        template_path = Path(data["templates"])
        value = cls(template_path if template_path.is_absolute() else root/template_path,
                    str(data["roi"]), HSVRange.from_mapping(data["white"]),
                    HSVRange.from_mapping(data["gold"]),
                    float(data["minimum_height_fraction"]), float(data["minimum_area_fraction"]),
                    float(data["baseline_tolerance"]), float(data["base_gap_ratio"]),
                    tuple(data["normalized_size"]), float(data["maximum_distance"]),
                    float(data["minimum_margin"]), data["confirmations"],
                    float(data["quantization_margin_seconds"]), float(data["maximum_sample_gap_seconds"]))
        fractions = (value.minimum_height_fraction, value.minimum_area_fraction,
                     value.baseline_tolerance, value.base_gap_ratio,
                     value.maximum_distance, value.minimum_margin)
        if (any(not math.isfinite(x) or not 0 < x < 1 for x in fractions) or
                type(value.confirmations) is not int or value.confirmations < 1 or
                len(value.normalized_size) != 2 or any(type(x) is not int or x < 8 for x in value.normalized_size) or
                not math.isfinite(value.quantization_margin_seconds) or value.quantization_margin_seconds < 0 or
                not math.isfinite(value.maximum_sample_gap_seconds) or value.maximum_sample_gap_seconds <= 0):
            raise ValueError("Invalid countdown recognition or stability bounds")
        return value


@dataclass(frozen=True)
class CountdownReading:
    status: str
    mode: str | None = None
    base_seconds: int | None = None
    reserve_seconds: int | None = None
    reason: str | None = None
    glyphs: tuple = ()
    reader_version: str | None = None

    @property
    def total_seconds(self):
        if self.status != "complete":
            return None
        return (self.base_seconds or 0) + (self.reserve_seconds or 0)

    def to_dict(self):
        return {**asdict(self), "total_seconds": self.total_seconds}


class CountdownReader:
    def __init__(self, config, *, turn_seconds, reserve_seconds):
        self.config = config
        self.turn_seconds, self.reserve_seconds = turn_seconds, reserve_seconds
        self.display_turn_seconds=math.ceil(turn_seconds+config.quantization_margin_seconds)
        glyph_config = SimpleNamespace(glyph_normalized_size=config.normalized_size,
                                       glyph_maximum_distance=config.maximum_distance,
                                       glyph_minimum_margin=config.minimum_margin)
        self.classifier = GlyphTemplateClassifier(config.templates, glyph_config)
        if set("0123456789") | {"plus"} != set(self.classifier.labels):
            raise ValueError("Countdown requires reviewed templates for 0..9 and plus")
        files = [{"path": str(path.relative_to(config.templates)),
                  "sha256": sha256(path.read_bytes()).hexdigest()}
                 for path in sorted(config.templates.glob("*/*"))
                 if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")]
        configuration = asdict(config)
        configuration["templates"] = str(config.templates)
        self.manifest = {"schema_version": 1, "config": configuration,
                         "rule_bounds": {"base": turn_seconds, "reserve": reserve_seconds},
                         "display_bounds":{"base":self.display_turn_seconds,"reserve":reserve_seconds},
                         "templates": files}
        self.version = sha256(json.dumps(self.manifest, sort_keys=True, default=str).encode()).hexdigest()
        self.manifest["version_sha256"] = self.version

    @classmethod
    def load(cls, path, root, rules):
        return None if path is None else cls(CountdownConfig.load(path, root),
                                            turn_seconds=rules["turn_seconds"],
                                            reserve_seconds=rules["reserve_seconds"])

    def segment(self, image, roi):
        crop = extract_roi(image, roi, self.config.roi)
        pieces = []
        for color in ("white", "gold"):
            mask = hsv_mask(crop.image, getattr(self.config, color))
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            for x, y, width, height, area in stats[1:]:
                if (height < mask.shape[0]*self.config.minimum_height_fraction or
                        area < mask.size*self.config.minimum_area_fraction):
                    continue
                box = crop.map_box_to_screen(Box(int(x), int(y), int(x+width), int(y+height)))
                local = mask[y:y+height, x:x+width]
                prediction = self.classifier.classify(local, box,
                    allowed_labels=set("0123456789") | ({"plus"} if color == "gold" else set()))
                pieces.append({"color": color, **asdict(prediction),
                               "clipped": bool(x == 0 or y == 0 or x+width == mask.shape[1] or y+height == mask.shape[0])})
        return tuple(pieces)

    def read(self, image, roi):
        return replace(self._read(image, roi), reader_version=self.version)

    def _read(self, image, roi):
        def number(labels, bound):
            if (not labels or len(labels) > len(str(bound)) or
                    any(label is None or not label.isdigit() for label in labels) or
                    len(labels) > 1 and labels[0] == "0"):
                return None
            value = int("".join(labels))
            return value if value <= bound else None

        try:
            pieces = self.segment(image, roi)
        except (KeyError, ValueError):
            return CountdownReading("unknown", reason="countdown_roi_unavailable")
        if not pieces:
            return CountdownReading("absent")
        bases = [p for p in pieces if p['color']=='white' and p['label'] is not None
                 and not p['clipped'] and int(p['label'])<=self.display_turn_seconds]
        def unrelated_to_base(piece):
            # A right-side wall edge above the digit row is not a timer glyph.
            # Aligned unknown/clipped gold remains blocking: it may be a cut
            # reserve digit, so never silently reinterpret that clock as 5+0.
            if len(bases)!=1:return False
            box,base=piece['box'],bases[0]['box']
            return box['x1']>base['x2'] and (box['y2']<base['y1'] or box['y1']>base['y2'])
        groups = []
        for piece in sorted((p for p in pieces if p["color"] == "gold" and not unrelated_to_base(p)), key=lambda p: p["box"]["y2"]):
            box = piece["box"]
            height = box["y2"]-box["y1"]
            group = next((g for g in groups if abs(g[0]["box"]["y2"]-box["y2"]) <=
                          self.config.baseline_tolerance*max(height, g[0]["box"]["y2"]-g[0]["box"]["y1"])), None)
            if group is None:
                groups.append([piece])
            else:
                group.append(piece)
        readings = []
        for group in groups:
            group.sort(key=lambda p: p["box"]["x1"])
            labels = [p["label"] for p in group]
            if any(label is None or p["clipped"] for label, p in zip(labels, group)):
                continue
            if labels[0] == "plus" and len(labels) > 1 and all(label.isdigit() for label in labels[1:]):
                plus = group[0]["box"]
                bases = [p for p in pieces if p["color"] == "white" and p["box"]["x2"] < plus["x1"]]
                if not bases:
                    continue
                base = max(bases, key=lambda p: p["box"]["x2"])
                box = base["box"]
                height = box["y2"]-box["y1"]
                if (base["label"] is None or base["clipped"] or
                        plus["x1"]-box["x2"] > self.config.base_gap_ratio*height or
                        abs(box["y2"]-group[-1]["box"]["y2"]) > self.config.baseline_tolerance*height):
                    continue
                base_seconds = int(base["label"])
                reserve = number(labels[1:], self.reserve_seconds)
                if base_seconds <= self.display_turn_seconds and reserve is not None:
                    readings.append(CountdownReading("complete", "base_plus_reserve", base_seconds, reserve,
                                                     glyphs=tuple([base, *group])))
            elif all(label.isdigit() for label in labels):
                reserve = number(labels, self.reserve_seconds)
                if reserve is not None:
                    readings.append(CountdownReading("complete", "reserve_only", 0, reserve, glyphs=tuple(group)))
        if not groups:
            bases = [p for p in pieces if p["color"] == "white" and p["label"] is not None and not p["clipped"]]
            if len(bases) == 1 and int(bases[0]["label"]) <= self.display_turn_seconds:
                readings.append(CountdownReading("complete", "base_only", int(bases[0]["label"]), 0, glyphs=tuple(bases)))
        if len(readings) == 1:
            return readings[0]
        return CountdownReading("unknown", reason="ambiguous_or_unrecognized_countdown", glyphs=pieces)


class DecisionBudget:
    """Confirm an observed clock once, then only shorten its absolute deadline."""
    def __init__(self, *, turn_seconds, reserve_seconds, action_margin, confirmations=2,
                 quantization_margin=1., maximum_sample_gap=2., fallback_turn_seconds=None):
        self.turn_seconds, self.reserve_seconds = turn_seconds, reserve_seconds
        self.fallback_turn_seconds = (turn_seconds if fallback_turn_seconds is None
                                      else fallback_turn_seconds)
        if (isinstance(self.fallback_turn_seconds, bool) or
                not isinstance(self.fallback_turn_seconds, (int, float)) or
                not math.isfinite(self.fallback_turn_seconds) or
                not action_margin < self.fallback_turn_seconds <= turn_seconds):
            raise ValueError('Fallback turn time must exceed action margin and not exceed turn time')
        self.action_margin = action_margin
        self.confirmations, self.quantization_margin = confirmations, quantization_margin
        self.maximum_sample_gap = maximum_sample_gap
        self.reset()

    def reset(self):
        self.started_at = self.absolute_deadline = self.last_sample = self.last_timestamp = None
        self.samples = 0
        self.candidate_deadline = None

    def start(self, timestamp):
        if self.started_at is None:
            self.started_at = timestamp

    def update(self, reading, timestamp):
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return
        self.last_timestamp = timestamp
        data = reading.to_dict() if isinstance(reading, CountdownReading) else reading
        if not data or data.get("status") != "complete":
            self.last_sample = None
            self.samples = 0
            return
        base, reserve = data.get("base_seconds"), data.get("reserve_seconds")
        if (type(base) is not int or type(reserve) is not int or
                not 0 <= base <= math.ceil(self.turn_seconds+self.quantization_margin) or
                not 0 <= reserve <= self.reserve_seconds):
            self.last_sample = None
            self.samples = 0
            return
        total = base + reserve
        candidate = timestamp+total-self.quantization_margin-self.action_margin
        if total == 0:
            self.absolute_deadline = min(self.absolute_deadline or candidate, candidate)
            return
        consistent = False
        if self.last_sample is not None:
            old_time, old_total = self.last_sample
            elapsed = timestamp-old_time
            consistent = (elapsed <= self.maximum_sample_gap and
                          0 <= old_total-total <= math.ceil(elapsed)+1)
        if consistent:
            self.samples += 1
            self.candidate_deadline = min(self.candidate_deadline, candidate)
        else:
            self.samples = 1
            self.candidate_deadline = (candidate if self.candidate_deadline is None
                                       else min(self.candidate_deadline, candidate))
        self.last_sample = (timestamp, total)
        if self.samples >= self.confirmations:
            self.absolute_deadline = (self.candidate_deadline if self.absolute_deadline is None
                                      else min(self.absolute_deadline, self.candidate_deadline))

    @property
    def pending(self):
        return self.absolute_deadline is None and 0 < self.samples < self.confirmations

    def deadline(self, captured_at, max_age):
        if self.started_at is None:
            self.start(captured_at)
        end = (self.absolute_deadline if self.absolute_deadline is not None
               else self.started_at+self.fallback_turn_seconds-self.action_margin)
        return min(end, captured_at+max_age)

    def to_dict(self):
        return {"source": "confirmed_visual_countdown" if self.absolute_deadline is not None else "base_time_fallback",
                "started_at": self.started_at, "absolute_deadline": self.absolute_deadline,
                "confirmation_pending": self.pending, "samples": self.samples,
                "fallback_turn_seconds": self.fallback_turn_seconds,
                "quantization_margin_seconds": self.quantization_margin}
