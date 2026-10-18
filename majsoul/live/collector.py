"""Auto-play data collector: the client plays by itself via its built-in
auto toggles; we capture every frame in the background (PrintWindow), keep the
toggle panel enabled each round, click through result screens, and archive key
frames plus tracker events for dataset building."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np
import yaml

from ..calibration.frames import write_image
from ..calibration.roi_config import ROIConfig
from ..perception import GlyphTemplateClassifier, SeatLayoutResolver, VisionConfig
from ..state import FrameObservationReader, MahjongStateTracker
from ..window import WindowError
from .capture import FocusUnavailable, LiveFrame, LiveWindow, roi_for_frame
from .classifier import build_tile_classifier
from .config import ROOT, LiveConfig
from .menu import MenuConfig, MenuNavigator
from .notify import FeishuNotifier
from .notifications import NotificationQueue
from .results import ResultDelivery
from .lifecycle import RoundLifecycle, SessionLifecycle
from .reference_archive import ReferenceArchive
from .buttons import ActionButtonReader


class CollectError(RuntimeError):
    pass


TOGGLE_LABELS = ("auto_sort", "auto_win", "no_call", "auto_discard", "auto_kita")
TOGGLE_LABELS_ZH = {"auto_sort": "Auto-sort hand", "auto_win": "Auto-win",
                    "no_call": "No chi/pon/kan", "auto_discard": "Auto draw-discard",
                    "auto_kita": "Auto nuki North"}


def _positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _point(name: str, value) -> tuple[float, float]:
    x, y = value
    x, y = float(x), float(y)
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError(f"{name} must be normalized to (0, 1)")
    return x, y


def _region(name: str, value) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in value)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"{name} must be a normalized [x1, y1, x2, y2] region")
    return x1, y1, x2, y2


def _resolve(root: Path, value) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class CollectConfig:
    live: Path
    dataset_root: Path
    poll_seconds: float
    triangle_point: tuple[float, float]
    triangle_open_point: tuple[float, float]
    toggle_x: float
    toggle_rows: dict[str, float]
    toggle_targets: tuple[str, ...]
    dot_scan_region: tuple[float, float, float, float]
    confirm_region: tuple[float, float, float, float]
    confirm_hold_seconds: float
    menu_template: Path
    menu_region: tuple[float, float, float, float]
    menu_threshold: float
    table_timeout_seconds: float
    end_screen_timeout_seconds: float
    calibration_size: tuple[int, int] = (2582, 1550)
    table_template: Path = ROOT / "artifacts/templates/collect/closed_tab.png"
    drawer_template: Path = ROOT / "artifacts/templates/collect/open_tab.png"
    table_region: tuple[float, float, float, float] = (0.0, 0.35, 0.10, 0.90)
    drawer_region: tuple[float, float, float, float] = (0.12, 0.40, 0.20, 0.85)
    ui_threshold: float = 0.85
    ui_maximum_error: float = 0.15
    table_ready_seconds: float = 1.0
    toggle_retry_seconds: float = 3.0
    toggle_max_attempts: int = 3
    toggle_settle_seconds: float = 0.35
    toggle_stable_samples: int = 2
    toggle_observation_attempts: int = 6
    toggle_sample_seconds: float = 0.15
    reference_before_seconds: float = 3.0
    reference_after_seconds: float = 3.0
    reference_sample_seconds: float = 0.5
    reference_maximum_frames: int = 32
    confirm_click_seconds: float = 2.5
    confirm_focus_attempts: int = 3
    unknown_timeout_seconds: float = 120.0
    match_result_templates: tuple[Path, ...] = ()
    match_result_region: tuple[float, float, float, float] = (0.3, 0.15, 0.65, 0.75)
    match_result_threshold: float = 0.85
    perception_interval_seconds: float = 2.0
    visual_change_threshold: float = 0.025
    keyframe_min_seconds: float = 0.5
    heartbeat_seconds: float = 30.0
    image_sample_size: tuple[int, int] = (320, 192)
    visual_change_grid: tuple[int, int] = (32, 24)
    dot_tolerance: float = 0.018
    off_dot_hsv_lower: tuple[int, int, int] = (100, 50, 40)
    off_dot_hsv_upper: tuple[int, int, int] = (125, 220, 170)
    dot_state_fraction: float = 0.6
    confirm_detection: dict = field(default_factory=lambda: {
        "gold_lower": [15, 101, 151], "gold_upper": [40, 255, 255],
        "blue_lower": [90, 81, 81], "blue_upper": [115, 255, 255],
        "gold_body_fraction": 0.3, "blue_body_fraction": 0.3,
        "gold_text_fraction": 0.02})
    dot_detection: dict = field(default_factory=lambda: {
        "green_min": 140, "green_blue_delta": 30, "green_red_delta": 40,
        "area_fraction": [250 / (2582 * 1550), 2500 / (2582 * 1550)],
        "aspect_ratio": [0.5, 2.0], "minimum_fill": 0.62})

    @classmethod
    def load(cls, path: str | Path, *, root: Path = ROOT) -> "CollectConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Malformed collect configuration")
        rows = data.get("toggle_rows")
        if not isinstance(rows, dict) or set(rows) != set(TOGGLE_LABELS):
            raise ValueError(
                "toggle_rows must map exactly: " + ", ".join(TOGGLE_LABELS))
        rows = {str(key): float(value) for key, value in rows.items()}
        for key, value in rows.items():
            if not 0 <= value <= 1:
                raise ValueError(f"toggle_rows.{key} must be normalized to (0, 1)")
        targets = tuple(str(item) for item in data.get(
            "toggle_targets", ("auto_win", "no_call", "auto_discard", "auto_kita")))
        unknown = set(targets) - set(TOGGLE_LABELS)
        if unknown:
            raise ValueError(f"unknown toggle_targets: {', '.join(sorted(unknown))}")
        template = _resolve(root, data.get(
            "menu_template", "artifacts/templates/menu/ranked_banner.png"))
        if not template.is_file():
            raise ValueError(f"menu_template does not exist: {template}")
        optional = {}
        result_paths = data.get("match_result_templates", [])
        if not isinstance(result_paths, list):
            raise ValueError("match_result_templates must be a list")
        optional["match_result_templates"] = tuple(_resolve(root, p) for p in result_paths)
        if any(not p.is_file() for p in optional["match_result_templates"]):
            raise ValueError("match_result_templates contains a missing file")
        for name in ("table_template", "drawer_template"):
            optional[name] = _resolve(root, data.get(
                name, f"artifacts/templates/collect/{'closed_tab' if name == 'table_template' else 'open_tab'}.png"))
            if not optional[name].is_file():
                raise ValueError(f"{name} does not exist: {optional[name]}")
        for name in ("table_region", "drawer_region", "match_result_region"):
            if name in data:
                optional[name] = _region(name, data[name])
        for name in ("table_ready_seconds", "toggle_retry_seconds", "toggle_settle_seconds", "toggle_sample_seconds",
                     "reference_before_seconds", "reference_after_seconds", "reference_sample_seconds",
                     "confirm_click_seconds", "unknown_timeout_seconds",
                     "perception_interval_seconds", "keyframe_min_seconds", "heartbeat_seconds"):
            if name in data:
                optional[name] = _positive(name, data[name])
        for name in ("ui_threshold", "ui_maximum_error", "visual_change_threshold", "dot_tolerance", "dot_state_fraction", "match_result_threshold"):
            if name in data:
                value = _positive(name, data[name])
                if value >= 1:
                    raise ValueError(f"{name} must be less than 1")
                optional[name] = value
        for name in ("off_dot_hsv_lower", "off_dot_hsv_upper"):
            if name in data:
                values = tuple(data[name])
                if len(values) != 3 or any(type(v) is not int or not 0 <= v <= (179 if i == 0 else 255)
                                           for i, v in enumerate(values)):
                    raise ValueError(f"{name} must be an HSV triple")
                optional[name] = values
        for name in ("calibration_size", "image_sample_size", "visual_change_grid"):
            if name in data:
                size = tuple(data[name])
                if len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
                    raise ValueError(f"{name} must be two positive integers")
                optional[name] = size
        if "toggle_max_attempts" in data:
            attempts = data["toggle_max_attempts"]
            if type(attempts) is not int or attempts < 1:
                raise ValueError("toggle_max_attempts must be a positive integer")
            optional["toggle_max_attempts"] = attempts
        for name in ("toggle_stable_samples", "toggle_observation_attempts"):
            value = data.get(name, cls.__dataclass_fields__[name].default)
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer of at least 2")
            optional[name] = value
        if optional["toggle_observation_attempts"] < optional["toggle_stable_samples"]:
            raise ValueError("toggle_observation_attempts must cover toggle_stable_samples")
        maximum = data.get("reference_maximum_frames", 32)
        if type(maximum) is not int or maximum < 2:
            raise ValueError("reference_maximum_frames must be an integer of at least 2")
        optional["reference_maximum_frames"] = maximum
        focus_attempts = data.get("confirm_focus_attempts", 3)
        if type(focus_attempts) is not int or focus_attempts < 1:
            raise ValueError("confirm_focus_attempts must be a positive integer")
        optional["confirm_focus_attempts"] = focus_attempts
        if "dot_detection" in data:
            settings = cls.__dataclass_fields__["dot_detection"].default_factory()
            if not isinstance(data["dot_detection"], dict) or set(data["dot_detection"]) - set(settings):
                raise ValueError("Malformed dot_detection")
            settings.update(data["dot_detection"])
            for name in ("green_min", "green_blue_delta", "green_red_delta"):
                if not 0 <= settings[name] <= 255:
                    raise ValueError(f"dot_detection.{name} must be in [0,255]")
            for name in ("area_fraction", "aspect_ratio"):
                lo, hi = settings[name]
                if not 0 < lo < hi or not math.isfinite(hi):
                    raise ValueError(f"dot_detection.{name} is invalid")
            if not 0 < settings["minimum_fill"] <= 1:
                raise ValueError("dot_detection.minimum_fill is invalid")
            optional["dot_detection"] = settings
        if "confirm_detection" in data:
            settings = cls.__dataclass_fields__["confirm_detection"].default_factory()
            raw = data["confirm_detection"]
            if not isinstance(raw, dict) or set(raw) - set(settings):
                raise ValueError("Malformed confirm_detection")
            settings.update(raw)
            for key, value in settings.items():
                if key.endswith("fraction"):
                    if not 0 < value <= 1:
                        raise ValueError(f"confirm_detection.{key} must be in (0,1]")
                elif len(value) != 3 or any(type(v) is not int or not 0 <= v <= (179 if i == 0 else 255)
                                            for i, v in enumerate(value)):
                    raise ValueError(f"confirm_detection.{key} must be an HSV triple")
            optional["confirm_detection"] = settings
        toggle_x = float(data.get("toggle_x", 0.0356))
        if not 0 < toggle_x < 1:
            raise ValueError("toggle_x must be normalized")
        threshold = float(data.get("menu_threshold", 0.55))
        if not 0 < threshold <= 1:
            raise ValueError("menu_threshold must be in (0,1]")
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("toggle_targets must be nonempty and unique")
        return cls(
            live=_resolve(root, data.get("live_config", "configs/live.steam.yaml")),
            dataset_root=_resolve(root, data.get("dataset_root", r"E:\majsoul_dataset")),
            poll_seconds=_positive("poll_seconds", data.get("poll_seconds", 0.5)),
            triangle_point=_point("triangle_point",
                                  data.get("triangle_point", (0.025, 0.761))),
            triangle_open_point=_point(
                "triangle_open_point",
                data.get("triangle_open_point", (0.155, 0.626))),
            toggle_x=toggle_x,
            toggle_rows=rows,
            toggle_targets=targets,
            dot_scan_region=_region("dot_scan_region", data.get(
                "dot_scan_region", (0.005, 0.30, 0.08, 0.92))),
            confirm_region=_region("confirm_region", data.get(
                "confirm_region", (0.84, 0.85, 0.97, 0.94))),
            confirm_hold_seconds=_positive(
                "confirm_hold_seconds", data.get("confirm_hold_seconds", 8)),
            menu_template=template,
            menu_region=_region("menu_region", data.get(
                "menu_region", (0.45, 0.12, 0.90, 0.50))),
            menu_threshold=threshold,
            table_timeout_seconds=_positive(
                "table_timeout_seconds", data.get("table_timeout_seconds", 300)),
            end_screen_timeout_seconds=_positive(
                "end_screen_timeout_seconds",
                data.get("end_screen_timeout_seconds", 120)),
            **optional,
        )


# ------------------------------------------------------------------ detection

def _crop_norm(image: np.ndarray, region: tuple[float, float, float, float]
               ) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = region
    box = (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))
    return image[box[1]:box[3], box[0]:box[2]], (box[0], box[1])


def find_on_dots(image: np.ndarray,
                 region: tuple[float, float, float, float],
                 settings: dict | None = None) -> list[float]:
    """Window-normalized y positions of green ON dots inside the given region.

    The drawer sits next to the vertical 理/和/鸣/切/拔 glyph strip; strokes of
    those glyphs pass a plain green mask, so components must also look like a
    filled disc (fill ratio close to pi/4, glyph strokes stay below ~0.45)."""
    settings = settings or CollectConfig.__dataclass_fields__["dot_detection"].default_factory()
    crop, (ox, oy) = _crop_norm(image, region)
    b, g, r = crop[:, :, 0].astype(int), crop[:, :, 1].astype(int), crop[:, :, 2].astype(int)
    mask = ((g > settings["green_min"]) &
            (g - b > settings["green_blue_delta"]) &
            (g - r > settings["green_red_delta"])).astype(np.uint8) * 255
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    lo, hi = (v * image.shape[0] * image.shape[1] for v in settings["area_fraction"])
    ys = []
    for index in range(1, count):
        w, h, area = stats[index, 2], stats[index, 3], stats[index, 4]
        if not lo <= area <= hi or not settings["aspect_ratio"][0] <= w / max(h, 1) <= settings["aspect_ratio"][1]:
            continue
        if area / (w * h) < settings["minimum_fill"]:
            continue
        ys.append((oy + centroids[index][1]) / image.shape[0])
    return sorted(ys)


def detect_confirm(image: np.ndarray,
                   region: tuple[float, float, float, float],
                   settings: dict | None = None) -> bool:
    """Confirm button presence. Two variants share the bottom-right slot:
    the end-of-match gold button (yellow body) and the in-round auto-dismissing
    blue button with gold text. The table felt is blue too, so the blue variant
    is recognized by its gold text, not its body."""
    return confirm_kind(image, region, settings) is not None


def confirm_kind(image: np.ndarray, region, settings: dict | None = None) -> str | None:
    settings = settings or CollectConfig.__dataclass_fields__["confirm_detection"].default_factory()
    crop, _ = _crop_norm(image, region)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gold = (cv2.inRange(hsv, np.array(settings["gold_lower"]),
                        np.array(settings["gold_upper"])) > 0).mean()
    if gold > settings["gold_body_fraction"]:
        return "gold"
    blue = (cv2.inRange(hsv, np.array(settings["blue_lower"]),
                        np.array(settings["blue_upper"])) > 0).mean()
    if blue > settings["blue_body_fraction"] and gold > settings["gold_text_fraction"]:
        return "blue"
    return None


def menu_score(image: np.ndarray, template: np.ndarray,
               region: tuple[float, float, float, float],
               reference_size: tuple[int, int] | None = None) -> float:
    if reference_size is not None:
        template = cv2.resize(template, (
            max(1, round(template.shape[1] * image.shape[1] / reference_size[0])),
            max(1, round(template.shape[0] * image.shape[0] / reference_size[1]))))
    crop, _ = _crop_norm(image, region)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    templ = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < templ.shape[0] or gray.shape[1] < templ.shape[1]:
        return 0.0
    return float(cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED).max())


class CollectionUI:
    """Only recognize UI anchors; tile/OCR failures cannot block auto-play."""

    def __init__(self, config: CollectConfig):
        self.config = config
        self.templates = {}
        for name, path in (("table", config.table_template),
                           ("drawer", config.drawer_template)):
            image = cv2.imread(str(path))
            if image is None:
                raise CollectError(f"Cannot read {name} template: {path}")
            self.templates[name] = image

    def visible(self, image: np.ndarray, name: str) -> bool:
        config = self.config
        crop, _ = _crop_norm(image, getattr(config, f"{name}_region"))
        template = self.templates[name]
        template = cv2.resize(template, (
            max(1, round(template.shape[1] * image.shape[1] / config.calibration_size[0])),
            max(1, round(template.shape[0] * image.shape[0] / config.calibration_size[1]))))
        if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
            return False
        scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if score < config.ui_threshold:
            return False
        patch = crop[y:y + template.shape[0], x:x + template.shape[1]]
        error = np.abs(patch.astype(np.float32) - template).mean() / max(float(template.mean()), 1.0)
        # Correlation alone accepts a dimmed result overlay as a live table.
        return bool(error <= config.ui_maximum_error)

    def table_visible(self, image: np.ndarray) -> bool:
        return self.visible(image, "table") or self.visible(image, "drawer")

    def confirm(self, image: np.ndarray) -> str | None:
        return confirm_kind(image, self.config.confirm_region,
                            self.config.confirm_detection)

    def match_result(self, image: np.ndarray) -> bool:
        paths = self.config.match_result_templates
        if not paths:
            return False
        if not hasattr(self, "_result_templates"):
            self._result_templates = tuple(cv2.imread(str(path)) for path in paths)
            if any(template is None for template in self._result_templates):
                raise CollectError("Cannot read final-result templates")
        return all(menu_score(image, template, self.config.match_result_region,
                              self.config.calibration_size) >= self.config.match_result_threshold
                   for template in self._result_templates)


# -------------------------------------------------------------- toggle panel

class TogglePanel:
    """Drive the left auto-play drawer: open it, enable target toggles, close."""

    def __init__(self, window: LiveWindow, config: CollectConfig,
                 *, sleep: Callable[[float], None] = time.sleep,
                 recorder: Callable[[LiveFrame, str], Path] | None = None,
                 capture: Callable[[], LiveFrame] | None = None):
        self.window = window
        self.config = config
        self.sleep = sleep
        self.ui = CollectionUI(config)
        self.recorder = recorder
        self.capture = capture or window.capture

    def _norm_click(self, point: tuple[float, float], frame: LiveFrame) -> None:
        self.window.click((point[0] * frame.window[2], point[1] * frame.window[3]),
                          expected=frame)

    def _row_targets(self, dots: list[float]) -> dict[str, tuple[float, float]]:
        rows = self.config.toggle_rows
        offset = 0.0
        anchors = [dot for dot in dots
                   if abs(dot - rows["auto_sort"]) <= self.config.dot_tolerance]
        if len(anchors) == 1:
            offset = anchors[0] - rows["auto_sort"]
        return {label: (self.config.toggle_x, rows[label] + offset)
                for label in self.config.toggle_targets}

    def missing(self, dots: list[float]) -> list[str]:
        targets = self._row_targets(dots)
        return [label for label, (_, y) in targets.items()
                if not any(abs(y - dot) <= self.config.dot_tolerance for dot in dots)]

    def _await_drawer(self) -> LiveFrame | None:
        """Bounded wait for the drawer animation; None if it never appeared."""
        for attempt in range(self.config.toggle_observation_attempts):
            if attempt:
                self.sleep(self.config.toggle_sample_seconds)
            frame = self.capture()
            if self.ui.confirm(frame.image):
                raise CollectError("result appeared while setting auto options")
            if self.ui.visible(frame.image, "drawer"):
                return frame
        return None

    def _read(self) -> tuple[LiveFrame, list[float]]:
        frame = self.capture()
        if self.ui.confirm(frame.image):
            raise CollectError("result appeared while setting auto options")
        if not self.ui.visible(frame.image, "drawer"):
            raise CollectError("toggle drawer is not visibly open")
        return frame, find_on_dots(frame.image, self.config.dot_scan_region,
                                   self.config.dot_detection)

    def state(self, image: np.ndarray, point: tuple[float, float]) -> bool | None:
        """Classify the same local switch interior as ON, OFF, or occluded."""
        x, y = round(point[0] * image.shape[1]), round(point[1] * image.shape[0])
        radius = max(1, int(math.sqrt(self.config.dot_detection["area_fraction"][0]
                                     * image.shape[0] * image.shape[1] / math.pi) / 2))
        patch = image[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
        if patch.size == 0:
            return None
        b, g, r = (patch[:, :, i].astype(int) for i in range(3))
        settings = self.config.dot_detection
        on = ((g > settings["green_min"]) &
              (g-b > settings["green_blue_delta"]) &
              (g-r > settings["green_red_delta"])).mean()
        off = (cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV),
                          np.array(self.config.off_dot_hsv_lower),
                          np.array(self.config.off_dot_hsv_upper)) > 0).mean()
        on, off = on >= self.config.dot_state_fraction, off >= self.config.dot_state_fraction
        return bool(on) if on != off else None

    def _stable_read(self, labels=None):
        """Bounded observation only; unknown/alternating states never authorize clicks."""
        labels = tuple(labels or self.config.toggle_targets)
        previous = dict.fromkeys(labels)
        counts = dict.fromkeys(labels, 0)
        states = dict.fromkeys(labels)
        for attempt in range(self.config.toggle_observation_attempts):
            if attempt:
                self.sleep(self.config.toggle_sample_seconds)
            frame, dots = self._read()
            points = self._row_targets(dots)
            for label in labels:
                current = self.state(frame.image, points[label])
                counts[label] = (counts[label]+1 if current is not None and current == previous[label]
                                 else int(current is not None))
                previous[label] = current
                states[label] = current if counts[label] >= self.config.toggle_stable_samples else None
            if all(state is not None for state in states.values()):
                break
        return frame, points, states

    def ensure(self, *, enabled: bool = True) -> list[str]:
        """Set the configured target toggles to a verified state."""
        frame = self.capture()
        if self.ui.confirm(frame.image):
            raise CollectError("result screen is visible; do not toggle")
        if not self.ui.visible(frame.image, "drawer"):
            if not self.ui.visible(frame.image, "table"):
                raise CollectError("table auto-option strip is not visible")
            self._norm_click(self.config.triangle_point, frame)
            self.sleep(self.config.toggle_settle_seconds)
            if self._await_drawer() is None:
                # Every sample showed the drawer closed, so the first click had
                # no effect; one more click cannot close an opening drawer.
                frame = self.capture()
                if self.ui.confirm(frame.image) or not self.ui.visible(frame.image, "table"):
                    raise CollectError("table changed while opening the toggle drawer")
                self._norm_click(self.config.triangle_point, frame)
                self.sleep(self.config.toggle_settle_seconds)
                self._await_drawer()
        try:
            # Reobserve every click: toggles invert, so blind retries can turn ON
            # back OFF when the UI is delayed or a new result screen appears.
            for label in self.config.toggle_targets:
                frame, points, states = self._stable_read((label,))
                if states[label] is not None and states[label] != enabled:
                    self._norm_click(points[label], frame)
                    self.sleep(self.config.toggle_settle_seconds)
                    self._stable_read((label,))
            frame, _, states = self._stable_read()
            missing = [label for label, state in states.items() if state != enabled]
            if self.recorder is not None:
                self.recorder(frame, "toggle_panel_incomplete" if missing else "toggle_panel_verified")
            return missing
        finally:
            frame = self.capture()
            if (not self.ui.confirm(frame.image)
                    and self.ui.visible(frame.image, "drawer")):
                self._norm_click(self.config.triangle_open_point, frame)
                self.sleep(self.config.toggle_settle_seconds)


# ----------------------------------------------------------------- collector

class MatchCollector:
    def __init__(self, window: LiveWindow, config: CollectConfig,
                 live: LiveConfig, *, max_matches: int = 0,
                 session: str | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], float] = time.monotonic,
                 notifier: FeishuNotifier | None = None):
        self.window = window
        self.config = config
        self.live = live
        if type(max_matches) is not int or max_matches < 0:
            raise ValueError("max_matches must be a nonnegative integer")
        self.max_matches = max_matches
        self.sleep = sleep
        self.now = now
        name = session or (time.strftime("collect-%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1000000000:09d}")
        if name in (".", "..") or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("session must be a single directory name")
        self.artifacts = config.dataset_root / name
        if self.artifacts.exists() and any(self.artifacts.iterdir()):
            raise CollectError("Session directory is not empty; use a new session name")
        self.notifier = notifier or FeishuNotifier(
            live.notify, logger=lambda message: self.log(
                "notify_error", error=message))
        self.notifications = (NotificationQueue(
            self.notifier, self.artifacts / "notifications",
            capacity=live.notify.queue_capacity, logger=self.log,
            max_attempts=live.notify.max_attempts, retry_seconds=live.notify.retry_seconds)
            if notifier is None else None)
        self._menu_template = cv2.imread(str(config.menu_template))
        if self._menu_template is None:
            raise CollectError(f"cannot read menu template: {config.menu_template}")
        self._menu_config = MenuConfig.load(live.menu)
        self._roi = ROIConfig.load(live.roi)
        self._roi_eff: ROIConfig | None = None
        self._roi_size: tuple[int, int] | None = None
        self._frame_index = 0
        self._seq = 0
        self._active_slots: tuple[str, ...] | None = None
        self._reader = None
        self._tracker = None
        self._resolver = None
        self.ui = CollectionUI(config)
        self._match_index = 0
        self._result_image = None
        self._result_delivery = None
        self._round_index = 0
        self._last_fp = None
        self._last_sample = None
        self._last_saved_at = float("-inf")
        self._last_perception_at = float("-inf")
        self._perception_failure = None
        self._button_reader = ActionButtonReader(live.buttons, live.button_templates)
        self._last_buttons = frozenset()
        self.references = ReferenceArchive(
            self.artifacts / "references", before_seconds=config.reference_before_seconds,
            after_seconds=config.reference_after_seconds,
            sample_seconds=config.reference_sample_seconds,
            maximum_frames=config.reference_maximum_frames)

    # ---------------------------------------------------------------- logging

    def log(self, event: str, **fields) -> None:
        entry = {"event": event, "wall_time": time.time(), "monotonic": self.now(),
                 "match_index": self._match_index, "round_index": self._round_index,
                 **fields}
        line = json.dumps(entry, ensure_ascii=False, default=str)
        print(line, flush=True)
        path = self.artifacts / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def _notify(self, message: str, image: Path | None = None, *, metadata=None):
        message = f"[majsoul-collect] {message}"
        if self.notifications is not None:
            return self.notifications.submit("notify_text", message, image, metadata=metadata)
        else:
            return self.notifier.notify_text(message, image)

    def _capture_result(self, frame):
        if self._result_image is not None or not self.ui.match_result(frame.image):
            return
        if self._result_delivery is None:
            self._result_delivery = ResultDelivery(
                self.artifacts, mode="collect", match_index=self._match_index,
                notify=self._notify, log=self.log)
        self._result_image = self._result_delivery.capture(
            frame, lambda result_frame: self._save_frame(result_frame, "match_result"))

    # ------------------------------------------------------------- primitives

    def _capture(self) -> LiveFrame:
        if (self.artifacts / "STOP").exists():
            raise KeyboardInterrupt
        frame = self.window.capture()
        self._frame_index += 1
        self.references.observe(frame, self._frame_index, self._match_index, self._round_index)
        return frame

    def _resolve_roi(self, frame: LiveFrame) -> ROIConfig:
        size = (int(frame.image.shape[1]), int(frame.image.shape[0]))
        if self._roi_size == size and self._roi_eff is not None:
            return self._roi_eff
        try:
            roi_eff = roi_for_frame(self._roi, size, self.live.projection)
        except ValueError as exc:
            raise CollectError(f"ROI projection unresolved: {exc}") from exc
        self._roi_eff = roi_eff
        self._roi_size = size
        self._resolver = None
        return roi_eff

    def _pipeline(self):
        if self._reader is None:
            vision = VisionConfig.load(self.live.vision)
            tile_classifier = build_tile_classifier(
                self.live.classifier, vision,
                cnn_checkpoint=self.live.cnn_checkpoint,
                template_root=self.live.tile_templates,
                expected_metadata=self.live.cnn_expected)
            manifest = getattr(tile_classifier, 'manifest', {'backend': self.live.classifier})
            (self.artifacts / 'classifier.json').write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            self.log('classifier_loaded', manifest=manifest)
            self._reader = FrameObservationReader(
                vision, tile_classifier,
                GlyphTemplateClassifier(self.live.glyph_templates, vision),
                button_reader=self._button_reader,
                read_scores=self.live.read_scores)
        return self._reader

    def _new_tracker(self) -> None:
        self._tracker = MahjongStateTracker.load(self.live.rules,
                                                 stable_frames=self.live.stable_frames)
        self._resolver = None
        self._active_slots = None
        self._last_fp = None
        self._last_sample = None
        self._last_perception_at = float("-inf")

    def _layout_resolver(self, roi_eff: ROIConfig) -> SeatLayoutResolver:
        if self._resolver is None:
            self._resolver = SeatLayoutResolver(
                roi_eff, self._pipeline().metadata_reader,
                starting_points=self._tracker.validator.starting_points,
                confirmations_required=self.live.layout_confirmations)
        return self._resolver

    # ------------------------------------------------------------------- run

    def run(self) -> int:
        self.window.restore_if_minimized()
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.log("session_start", session=self.artifacts.name,
                 dataset_root=str(self.config.dataset_root),
                 max_matches=self.max_matches)
        (self.artifacts / "session.json").write_text(json.dumps({
            "schema_version": 1, "source": "steam_live_builtin_autoplay",
            "player_count": self._roi.player_count, "mode": "sanma_east",
            "capture": "PrintWindow", "input": "temporary_foreground",
            "started_at": time.time(), "toggle_targets": self.config.toggle_targets,
            "live_config": str(self.config.live),
            "calibration_size": self.config.calibration_size,
            "labels_verified": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        completed = 0
        try:
            while True:
                self._match_index = completed + 1
                self._round_index = 0
                self._result_image = None
                self._result_delivery = None
                self._enter_match()
                # _play_match only finishes after the lobby is visually verified.
                self._play_match()
                completed += 1
                self.log("match_complete", matches=completed)
                self._notify(f"number {completed} sanma East matches completed, data saved in {self.artifacts.name}"
                             + ("; final result screenshot has been submitted separately" if self._result_image else "; final result screenshot not obtained"))
                if self.max_matches and completed >= self.max_matches:
                    self.log("session_end", matches=completed)
                    return 0
        except KeyboardInterrupt:
            self.log("interrupted", matches=completed)
            return 130
        except (RuntimeError, ValueError, OSError) as exc:
            self.log("fatal_error", error=str(exc))
            self._notify(f"collection terminated abnormally: {exc}")
            raise
        finally:
            self.references.close()
            if self.notifications is not None:
                self.notifications.close(self.live.notify.shutdown_seconds)

    def _enter_match(self) -> None:
        self._new_tracker()
        probe = self._capture()
        self._resolve_roi(probe)
        if self.ui.table_visible(probe.image) or self.ui.confirm(probe.image):
            self.log("menu_skipped", reason="table or result visible")
            return
        score = self._menu_score(probe)
        if score >= self.config.menu_threshold and self._menu_config.ready:
            self._save_frame(probe, "menu_navigation_start")
            MenuNavigator(self.window, self._menu_config, sleep=self.sleep).run()
            self.log("menu_navigated")
        elif score >= self.config.menu_threshold:
            raise CollectError("Menu coordinates are not calibrated")
        else:
            self.log("waiting_for_table", reason="unrecognized screen; no blind menu clicks")
        deadline = self.now() + self.config.table_timeout_seconds
        while self.now() < deadline:
            frame = self._capture()
            self._resolve_roi(frame)
            self._visual_record(frame)
            self._capture_result(frame)
            if self.ui.table_visible(frame.image) or self.ui.confirm(frame.image):
                self._save_frame(frame, "match_entered")
                return
            self.sleep(self.config.poll_seconds)
        path = self.artifacts / "table_timeout.jpg"
        write_image(path, frame.image)
        self._notify("waited for the table to time out, did not enter the match", path)
        raise CollectError("table screen not detected within "
                           f"{self.config.table_timeout_seconds:g}s")

    def _menu_score(self, frame: LiveFrame) -> float:
        return menu_score(frame.image, self._menu_template,
                          self.config.menu_region, self.config.calibration_size)

    # ------------------------------------------------------------- recording

    def _track(self, frame: LiveFrame, roi_eff: ROIConfig):
        try:
            if self._active_slots is None:
                resolution = self._layout_resolver(roi_eff).observe(frame.image)
                if resolution.locked_layout is None:
                    return None
                self._active_slots = tuple(resolution.locked_layout)
                self.log("layout_locked", active_slots=list(self._active_slots))
            observation = self._pipeline().read(
                frame.image, roi_eff, self._active_slots,
                timestamp=frame.captured_at, frame_index=self._frame_index)
            update = self._tracker.update(observation)
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            if str(exc) != self._perception_failure:
                self.log("perception_error", frame_index=self._frame_index,
                         error=str(exc))
                self._perception_failure = str(exc)
            return None
        self._perception_failure = None
        return observation, update

    def _record(self, frame: LiveFrame, observation, update) -> None:
        if not update.events and observation.fingerprint == self._last_fp:
            return
        self._save_frame(frame, "tracker_event" if update.events else "observation_changed",
                         observation, update)
        self._last_fp = observation.fingerprint

    def _save_frame(self, frame: LiveFrame, reason: str, observation=None, update=None,
                    *, reference_tags=()) -> Path:
        self._seq += 1
        name = f"{self._seq:08d}.jpg"
        path = self.artifacts / "frames" / name
        write_image(path, frame.image)
        events = [event.to_dict() for event in update.events] if update else []
        record = {
            "seq": self._seq,
            "file": f"frames/{name}",
            "reason": reason,
            "reference_tags": list(reference_tags),
            "wall_time": time.time(),
            "match_index": self._match_index, "round_index": self._round_index,
            "player_count": self._roi.player_count, "mode": "sanma_east",
            "active_slots": self._active_slots,
            "window": frame.window,
            "captured_at": frame.captured_at,
            "frame_index": self._frame_index,
            "fingerprint": observation.fingerprint if observation else None,
            "issues": list(observation.issues) if observation else ["semantic_observation_unavailable"],
            "events": events,
            "observation": observation.to_dict() if observation else None,
            "committed": bool(update and update.committed),
            "validation_issues": list(update.validation_issues) if update else [],
        }
        self.artifacts.mkdir(parents=True, exist_ok=True)
        with (self.artifacts / "observations.jsonl").open(
                "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if events:
            self.log("events", frame_index=self._frame_index, file=record["file"], events=events)
        tags = list(reference_tags) + ["event:" + event["type"] for event in events]
        if reason in {"match_result", "confirm_appeared", "confirm_disappeared", "table_ready",
                      "toggle_panel_incomplete", "end_screen_stuck", "unknown_screen_timeout"}:
            tags.append("capture:" + reason)
        if tags:
            self.references.trigger(tags, frame, self._frame_index, self._match_index, self._round_index)
        self._last_sample = cv2.resize(frame.image, self.config.image_sample_size)
        self._last_saved_at = self.now()
        return path

    def _visual_record(self, frame: LiveFrame) -> None:
        # A visible operation is useful evidence even while seat/CNN tracking is unavailable.
        roi = self._roi_eff or self._roi
        buttons = frozenset(self._button_reader.read(frame.image, roi))
        appeared = buttons - self._last_buttons
        self._last_buttons = buttons
        if appeared:
            self._save_frame(frame, "button_appeared",
                             reference_tags=["button:" + name for name in sorted(appeared)])
        sample = cv2.resize(frame.image, self.config.image_sample_size)
        elapsed = self.now() - self._last_saved_at
        changed = self._last_sample is None
        if not changed:
            difference = np.abs(sample.astype(np.float32) - self._last_sample).mean(axis=2) / 255
            mask = np.zeros(difference.shape, dtype=np.float32)
            roi = self._roi_eff or self._roi
            height, width = difference.shape
            for region in roi.regions.values():
                if region is not None:
                    x1, y1, x2, y2 = region
                    mask[int(y1*height):max(int(y1*height)+1, int(y2*height)),
                         int(x1*width):max(int(x1*width)+1, int(x2*width))] = 1
            # A single discarded tile occupies far too little of the whole
            # window to pass a global mean threshold. Pool local changes within
            # configured gameplay ROIs instead; this records evidence, not an
            # inferred mahjong event.
            blocks = cv2.resize(difference * mask, self.config.visual_change_grid,
                                 interpolation=cv2.INTER_AREA)
            changed = float(blocks.max()) >= self.config.visual_change_threshold
        if elapsed >= self.config.heartbeat_seconds or (changed and elapsed >= self.config.keyframe_min_seconds):
            self._save_frame(frame, "visual_change" if changed else "heartbeat")

    # ------------------------------------------------------------ match loop

    def _play_match(self) -> None:
        self._last_fp = None
        panel = TogglePanel(self.window, self.config, sleep=self.sleep,
                            recorder=self._save_frame, capture=self._capture)
        lifecycle = SessionLifecycle(self.config.table_ready_seconds)
        attempts = 0
        toggles_verified = False
        retry_at = float("-inf")
        next_confirm_click = float("-inf")
        focus_failures = 0
        ending_since = None
        last_known = self.now()
        menu_since = None
        minimized_logged = False
        while True:
            try:
                frame = self._capture()
            except WindowError as exc:
                if "minimized" in str(exc).lower():
                    if not minimized_logged:
                        self.log("window_minimized", note="waiting for restore")
                        minimized_logged = True
                    lifecycle.reset_stability()
                    self.sleep(self.config.poll_seconds)
                    continue
                raise
            if minimized_logged:
                last_known = self.now()
                ending_since = None
                menu_since = None
                self.log("window_restored")
                minimized_logged = False
            now = self.now()
            roi_eff = self._resolve_roi(frame)
            self._visual_record(frame)
            self._capture_result(frame)
            lobby = self._menu_score(frame) >= self.config.menu_threshold
            kind = self.ui.confirm(frame.image)
            table = not lobby and not kind and self.ui.table_visible(frame.image)
            transitions = lifecycle.update(confirm=kind is not None, table=table,
                                           lobby=lobby, now=now)
            if "match_complete" in transitions:
                if self._result_delivery is not None:
                    self._result_delivery.lobby_verified(frame)
                path = self._save_frame(frame, "menu_reached")
                self.log("menu_reached", file=str(path.relative_to(self.artifacts)))
                return
            if lobby:
                self.sleep(self.config.poll_seconds)
                continue
            for transition in transitions:
                if transition == "table_ready":
                    self._round_index = lifecycle.round_index
                    attempts = 0
                    toggles_verified = False
                    retry_at = now
                path = self._save_frame(frame, transition)
                self.log(transition, file=str(path.relative_to(self.artifacts)), confirm_kind=kind)
            if kind or table:
                last_known = now
            if kind:
                held = now - lifecycle.confirm_since if lifecycle.confirm_since is not None else 0
                # Normal blue confirms are left to the client's three-second
                # countdown. A gold confirm or a stuck countdown may need input.
                if ((kind == "gold" and held >= self.config.table_ready_seconds) or
                        held >= self.config.confirm_hold_seconds):
                    if ending_since is None:
                        ending_since = now
                        self.log("result_dismissal_started", kind=kind, hold_seconds=held)
                    if now >= next_confirm_click:
                        try:
                            self._click_confirm(frame)
                            focus_failures = 0
                        except FocusUnavailable as exc:
                            focus_failures += 1
                            self._save_frame(frame, "confirm_focus_unavailable")
                            self.log("confirm_focus_deferred", attempts=focus_failures, error=str(exc))
                            if focus_failures >= self.config.confirm_focus_attempts:
                                raise CollectError("Cannot acquire focus for confirmation after bounded retries") from exc
                        next_confirm_click = self.now() + self.config.confirm_click_seconds
            elif table:
                # A persistent confirm can belong to a hand, so only a verified
                # lobby completes a match. Returning to a table resumes play.
                ending_since = None
            if (table and not lifecycle.pending and not toggles_verified
                    and attempts < self.config.toggle_max_attempts and now >= retry_at):
                attempts += 1
                missing = self._setup_toggles(panel)
                toggles_verified = not missing
                retry_at = self.now() + self.config.toggle_retry_seconds
                path = self._save_frame(self._capture(), "toggle_verified" if toggles_verified else "toggle_retry")
                if missing and attempts >= self.config.toggle_max_attempts:
                    zh = ", ".join(TOGGLE_LABELS_ZH[label] for label in missing)
                    self.log("toggle_unverified", missing=missing, attempts=attempts)
                    self._notify(f"autoplay option activation was not confirmed: {zh}", path)
                    raise CollectError("auto options remain unverified after bounded retries")
                # Captures made before clicking are no longer current.
                self.sleep(self.config.poll_seconds)
                continue
            if table and toggles_verified and not lifecycle.pending and self._table_hook(frame):
                # The hook may have changed options; verify all of them again.
                toggles_verified = False
                attempts = 0
                retry_at = self.now()
                self.sleep(self.config.poll_seconds)
                continue
            if table and toggles_verified and now - self._last_perception_at >= self.config.perception_interval_seconds:
                self._last_perception_at = now
                tracked = self._track(frame, roi_eff)
                if tracked is not None:
                    self._record(frame, *tracked)
            if ending_since is not None and now - ending_since >= self.config.end_screen_timeout_seconds:
                path = self._save_frame(frame, "end_screen_stuck")
                self._notify("settlement confirmation timed out, has not returned to the main screen, manual takeover required", path)
                raise CollectError("result screens did not reach the lobby")
            if now - last_known >= self.config.unknown_timeout_seconds:
                path = self._save_frame(frame, "unknown_screen_timeout")
                self._notify("screen remains unrecognizable, capture paused, please check the window", path)
                raise CollectError("unrecognized screen timed out")
            self.sleep(self.config.poll_seconds)

    def _table_hook(self, frame: LiveFrame) -> bool:
        """Supervised-protection hook on a table frame with verified options.

        Return True when the auto options may no longer be in their verified
        state. Collection itself never changes them here."""
        return False

    def _setup_toggles(self, panel: TogglePanel) -> list[str]:
        self.log("toggle_setup_start")
        try:
            missing = panel.ensure()
        except (CollectError, WindowError) as exc:
            self.log("toggle_setup_error", error=str(exc))
            return list(self.config.toggle_targets)
        self.log("toggle_setup_done", missing=missing)
        return missing

    # ------------------------------------------------------------- match end

    def _click_confirm(self, frame: LiveFrame) -> bool:
        region = self.config.confirm_region
        point = ((region[0] + region[2]) / 2, (region[1] + region[3]) / 2)
        # Capture again immediately before focusing: result overlays can auto
        # dismiss between the polling frame and the click.
        current = self._capture()
        if not self.ui.confirm(current.image) or current.window != frame.window:
            self.log("end_confirm_cancelled", reason="result changed before click")
            return False
        path = self._save_frame(current, "end_confirm_before_click")
        # Disk I/O is kept outside the final observation-to-click interval.
        current = self._capture()
        if not self.ui.confirm(current.image) or current.window != frame.window:
            return False
        self.window.click((point[0] * current.window[2], point[1] * current.window[3]),
                          expected=current)
        self.log("end_confirm_clicked", file=str(path.relative_to(self.artifacts)))
        return True

    def _return_to_menu(self) -> bool:
        deadline = self.now() + self.config.end_screen_timeout_seconds
        clicks = 0
        while self.now() < deadline:
            frame = self._capture()
            score = self._menu_score(frame)
            if score >= self.config.menu_threshold:
                self.log("menu_reached", score=round(score, 3), clicks=clicks)
                return True
            if self.ui.confirm(frame.image):
                clicks += int(self._click_confirm(frame))
                self.sleep(self.config.confirm_click_seconds)
            else:
                self.sleep(1.0)
        path = self.artifacts / "end_screen_stuck.jpg"
        write_image(path, frame.image)
        self.log("end_screen_stuck", clicks=clicks)
        self._notify("failed to return to the main screen at match end, manual takeover required", path)
        return False
