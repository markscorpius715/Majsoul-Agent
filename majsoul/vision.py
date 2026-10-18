from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from statistics import median

from .domain import Box, Detection, MeldView, Observation, TileView
from .rules import Rules
from .tiles import tile_index


REQUIRED_REGIONS = ("hand", "draw", "dora", *(f"{kind}:{p}" for kind in ("river", "meld", "nuki") for p in range(3)))


@dataclass
class Layout:
    regions: dict
    labels: dict[str, str]
    confidence: float = 0.8
    proposal_confidence: float = 0.15
    duplicate_iou: float = 0.6
    stable_frames: int = 2
    max_decision_age: float = 10.0
    verification_timeout: float = 4.0
    poll_seconds: float = 0.2
    target_tolerance: float = 0.02
    calibration_notes: str = ""
    action_margin_seconds: float = 1.0

    @classmethod
    def load(cls, path: str | Path) -> "Layout":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        obj = cls(**data)
        if set(obj.regions) != set(REQUIRED_REGIONS):
            raise ValueError(f"Layout requires exactly these regions: {REQUIRED_REGIONS}")
        for name, region in obj.regions.items():
            rect = region.get("rect")
            if not isinstance(rect, list) or len(rect) != 4:
                raise ValueError(f"Region {name} has not been calibrated")
            x1, y1, x2, y2 = rect
            if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
                raise ValueError("Regions must be normalized nonempty rectangles")
            if region.get("row_axis", "y") not in ("x", "y"):
                raise ValueError("row_axis must be x or y")
            if region.get("row_direction", 1) not in (-1, 1) or region.get("tile_direction", 1) not in (-1, 1):
                raise ValueError("Sort directions must be -1 or 1")
        if not 0 < obj.proposal_confidence <= obj.confidence <= 1 or not 0 < obj.duplicate_iou <= 1:
            raise ValueError("Invalid confidence/IoU settings")
        if obj.stable_frames < 2 or any(not 0 < v < float("inf") for v in (
            obj.max_decision_age, obj.verification_timeout, obj.poll_seconds, obj.target_tolerance, obj.action_margin_seconds)):
            raise ValueError("Invalid timing/stability settings")
        return obj


def iou(a: Box, b: Box) -> float:
    intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    union = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - intersection
    return intersection / union if union > 0 else 0


def ordered(detections: list[Detection], region: dict) -> list[Detection]:
    """Cluster rows using object size, rather than sorting jittery y coordinates."""
    if not detections:
        return []
    axis = 1 if region.get("row_axis", "y") == "y" else 0
    width = median((d.box.y2 - d.box.y1) if axis else (d.box.x2 - d.box.x1) for d in detections)
    rows: list[list[Detection]] = []
    for detection in sorted(detections, key=lambda d: d.box.center[axis]):
        if not rows or detection.box.center[axis] - median(d.box.center[axis] for d in rows[-1]) > width / 2:
            rows.append([detection])
        else:
            rows[-1].append(detection)
    if region.get("row_direction", 1) < 0:
        rows.reverse()
    return [d for row in rows for d in sorted(row, key=lambda d: d.box.center[1-axis],
                                             reverse=region.get("tile_direction", 1) < 0)]


class YoloDetector:
    def __init__(self, weights: Path, layout: Layout, device: str = "cpu"):
        if not weights.is_file():
            raise FileNotFoundError(f"Custom mahjong detector weights missing: {weights}")
        settings_dir = Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / "artifacts" / "ultralytics")))
        settings_dir.mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.layout = layout
        self.device = device

    def detect(self, image) -> list[Detection]:
        result = self.model.predict(source=image, conf=self.layout.proposal_confidence,
                                    device=self.device, verbose=False)[0]
        if result.boxes is None:
            return []
        return [Detection(result.names[int(cls)], float(conf), Box(*map(float, xyxy)))
                for xyxy, conf, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]


class ObservationBuilder:
    def __init__(self, layout: Layout, rules: Rules):
        self.layout, self.rules = layout, rules

    def build(self, detections: list[Detection], captured_at: float, window: tuple[int, int, int, int]) -> Observation:
        _, _, width, height = window
        obs = Observation(captured_at, window)
        buckets: dict[str, list[Detection]] = {r: [] for r in REQUIRED_REGIONS}
        metadata: dict[str, str] = {}
        riichi = [False] * 3
        closed: set[tuple[int, int]] = set()
        accepted: list[Detection] = []
        for raw in detections:
            coords = (raw.box.x1, raw.box.y1, raw.box.x2, raw.box.y2, raw.confidence)
            if not all(math.isfinite(x) for x in coords) or not 0 <= raw.confidence <= 1:
                obs.issues.append("Invalid detector numbers")
                continue
            if not (0 <= raw.box.x1 < raw.box.x2 <= width and 0 <= raw.box.y1 < raw.box.y2 <= height):
                obs.issues.append("Detection outside captured client area")
                continue
            label = self.layout.labels.get(raw.label, raw.label)
            if label == "ignore":
                continue
            x, y = raw.box.center
            regions = [name for name, r in self.layout.regions.items()
                       if r["rect"][0] <= x / width < r["rect"][2] and r["rect"][1] <= y / height < r["rect"][3]]
            semantic = label.startswith(("tile:", "button:", "phase:", "meta:", "riichi:", "closed:"))
            if not semantic:
                if regions:
                    obs.issues.append(f"Unknown detection in monitored region: {label}")
                continue
            if raw.confidence < self.layout.confidence:
                obs.issues.append(f"Low confidence: {label}")
                continue
            if any(iou(raw.box, d.box) >= self.layout.duplicate_iou for d in accepted):
                obs.issues.append("Overlapping/ambiguous detections")
                continue
            accepted.append(raw)
            if label.startswith("tile:"):
                tile = label.removeprefix("tile:")
                try:
                    self.rules.validate_tiles([tile])
                except ValueError:
                    obs.issues.append(f"Invalid tile: {tile}")
                    continue
                if len(regions) != 1:
                    obs.issues.append("Tile belongs to zero or multiple configured regions")
                    continue
                buckets[regions[0]].append(Detection(tile, raw.confidence, raw.box))
            elif label.startswith("button:"):
                button = label.split(":", 1)[1]
                if button in obs.buttons:
                    obs.issues.append("Duplicate action button")
                obs.buttons[button] = raw.box
            elif label.startswith("phase:"):
                if obs.phase != "unknown":
                    obs.issues.append("Multiple turn phases")
                obs.phase = label.split(":", 1)[1]
            elif label.startswith("meta:"):
                _, key, value = label.split(":", 2)
                if key in metadata:
                    obs.issues.append(f"Duplicate metadata: {key}")
                metadata[key] = value
            elif label.startswith("riichi:"):
                player = int(label.split(":")[1])
                if player not in range(3):
                    raise ValueError("Invalid riichi player index")
                riichi[player] = True
            elif label.startswith("closed:"):
                _, player, tile = label.split(":")
                closed.add((int(player), tile_index(tile)))
        buckets = {name: ordered(ds, self.layout.regions[name]) for name, ds in buckets.items()}
        hand, draw = buckets["hand"], buckets["draw"]
        if len(draw) > 1:
            obs.issues.append("More than one drawn tile")
        obs.hand = tuple(TileView(d.label, d.box) for d in hand + draw)
        obs.draw_index = len(hand) if len(draw) == 1 else None
        obs.rivers = tuple(tuple(d.label for d in buckets[f"river:{p}"]) for p in range(3))
        melds = []
        for player in range(3):
            groups: dict[int, list[str]] = defaultdict(list)
            for d in buckets[f"meld:{player}"]:
                groups[tile_index(d.label)].append(d.label)
            if any(len(ts) not in (3, 4) for ts in groups.values()) or len(groups) > 4:
                obs.issues.append("Incomplete or impossible meld; hidden kan tiles need a CV adapter")
            melds.append(tuple(MeldView(tuple(ts), (player, index) not in closed) for index, ts in sorted(groups.items())))
        obs.melds = tuple(melds)
        obs.nuki = tuple(len(buckets[f"nuki:{p}"]) for p in range(3))
        if any(d.label != self.rules.data["nuki_tile"] for p in range(3) for d in buckets[f"nuki:{p}"]):
            obs.issues.append("Non-north tile in nuki area")
        obs.dora_indicators = tuple(d.label for d in buckets["dora"])
        obs.riichi = tuple(riichi)
        obs.seat_wind = metadata.get("seat")
        obs.round_wind = metadata.get("round")
        if all(metadata.get(k) for k in ("round", "number", "honba")):
            obs.round_id = ":".join(metadata[k] for k in ("round", "number", "honba"))
        if obs.phase != "result" and (not obs.round_id or obs.seat_wind not in ("1z", "2z", "3z") or obs.round_wind not in ("1z", "2z", "3z")):
            obs.issues.append("Missing/invalid round, hand number, honba or seat metadata")
        if obs.phase not in ("discard", "response", "riichi_select", "waiting", "result"):
            obs.issues.append("Unknown phase; call-selection dialogs require an adapter")
        if obs.phase != "result":
            expected = 13 - 3 * len(obs.melds[0])
            if len(obs.hand) not in (expected, expected + 1):
                obs.issues.append("Concealed hand count inconsistent with melds")
            if obs.phase in ("discard", "riichi_select") and len(obs.hand) != expected + 1:
                obs.issues.append("Discard phase requires the extra concealed tile")
            if obs.phase in ("response", "waiting") and len(obs.hand) != expected:
                obs.issues.append("Waiting/response phase must not contain an extra concealed tile")
            if not 1 <= len(obs.dora_indicators) <= 5:
                obs.issues.append("Missing/invalid dora indicator count")
            visible = ([t.tile for t in obs.hand] + [t for r in obs.rivers for t in r]
                       + [t for ms in obs.melds for m in ms for t in m.tiles]
                       + list(obs.dora_indicators) + [self.rules.data["nuki_tile"]] * sum(obs.nuki))
            try:
                self.rules.validate_tiles(visible)
            except ValueError as exc:
                obs.issues.append(str(exc))
        return obs
