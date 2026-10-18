"""Validated configuration for the live Steam-client agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re

import yaml

from ..calibration.roi_config import PROJECTION_NAMES
from .buttons import ButtonConfig
from .selection import RiichiSelectionConfig
from .riichi import RiichiConfig
from .emergency import EmergencyConfig
from .protection import ProtectionConfig
from .watchdog import WatchdogConfig

ROOT = Path(__file__).resolve().parents[2]

CLASSIFIER_KINDS = ("auto", "cnn", "template")
POLICY_KINDS = ("auto", "jev", "deepseek")


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _positive(name: str, value: float, *, upper: float | None = None,
              include_upper: bool = True) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if upper is not None and (result > upper if include_upper else result >= upper):
        bound = "<=" if include_upper else "<"
        raise ValueError(f"{name} must be {bound} {upper}")
    return result


@dataclass(frozen=True)
class DecisionConfig:
    policy: str = "auto"
    jev_proxy: str | None = None
    action_margin_seconds: float = 1.0
    max_decision_age: float = 10.0
    hedge_seconds: float = 1.0           # start the fallback model if the primary is still silent
    local_fallback_seconds: float = 0.8  # deadline reserve for the local efficiency rung; 0 disables
    always_win: bool = True              # a legal ron/tsumo is taken without asking the model
    shanten_guard: bool = True           # hide shanten-losing discards unless a riichi threat is visible

    @classmethod
    def from_dict(cls, data: dict | None) -> "DecisionConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("decision configuration must be a mapping")
        policy = str(data.get("policy", "auto"))
        if policy not in POLICY_KINDS:
            raise ValueError(f"decision.policy must be one of: {', '.join(POLICY_KINDS)}")
        proxy = data.get("jev_proxy")
        if proxy is not None:
            proxy = str(proxy)
            if not re.fullmatch(r"https?://[^\s/]+(:\d+)?/?", proxy):
                raise ValueError("decision.jev_proxy must be an http(s) host URL")
        margin = _positive("decision.action_margin_seconds",
                           data.get("action_margin_seconds", 1.0))
        age = _positive("decision.max_decision_age", data.get("max_decision_age", 10))
        hedge = _positive("decision.hedge_seconds", data.get("hedge_seconds", 1.0))
        local = data.get("local_fallback_seconds", 0.8)
        if type(local) not in (int, float) or not math.isfinite(local) or not 0 <= local < age:
            raise ValueError("decision.local_fallback_seconds must be in [0, max_decision_age)")
        always_win = data.get("always_win", True)
        if type(always_win) is not bool:
            raise ValueError("decision.always_win must be boolean")
        guard = data.get("shanten_guard", True)
        if type(guard) is not bool:
            raise ValueError("decision.shanten_guard must be boolean")
        return cls(policy, proxy, margin, age, hedge, float(local), always_win, guard)


@dataclass(frozen=True)
class ExecutionConfig:
    verification_timeout: float = 4.0
    target_tolerance: float = 0.02
    mouse_press_seconds: float = 0.05
    hand_click_count: int = 1
    inter_click_seconds: float = 0.05

    @classmethod
    def from_dict(cls, data: dict | None) -> "ExecutionConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("execution configuration must be a mapping")
        timeout = _positive("execution.verification_timeout",
                            data.get("verification_timeout", 4.0))
        tolerance = _positive("execution.target_tolerance",
                              data.get("target_tolerance", 0.02), upper=0.2)
        press = _positive('execution.mouse_press_seconds',
                          data.get('mouse_press_seconds', 0.05), upper=0.2)
        count = data.get('hand_click_count', 1)
        if type(count) is not int or count not in (1, 2):
            raise ValueError('execution.hand_click_count must be 1 or 2')
        interval = _positive('execution.inter_click_seconds',
                             data.get('inter_click_seconds', 0.05), upper=0.2)
        return cls(timeout, tolerance, press, count, interval)


@dataclass(frozen=True)
class NotifyConfig:
    enabled: bool = True
    feishu_helper: Path | None = None
    chat_id: str | None = None
    node: str = "node"
    timeout_seconds: float = 10.0
    queue_capacity: int = 32
    shutdown_seconds: float = 3.0
    proxy: str | None = None
    max_attempts: int = 1
    retry_seconds: float = 1.0

    @classmethod
    def from_dict(cls, data: dict | None, root: Path) -> "NotifyConfig":
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("notify configuration must be a mapping")
        enabled = data.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("notify.enabled must be boolean")
        node = str(data.get("node", "node"))
        if not node.strip():
            raise ValueError("notify.node must not be empty")
        helper = data.get("feishu_helper")
        chat_id = data.get("chat_id")
        timeout = _positive("notify.timeout_seconds", data.get("timeout_seconds", 10))
        shutdown = _positive("notify.shutdown_seconds", data.get("shutdown_seconds", 3))
        capacity = data.get("queue_capacity", 32)
        attempts = data.get("max_attempts", 1)
        retry = _positive("notify.retry_seconds", data.get("retry_seconds", 1))
        if type(attempts) is not int or attempts < 1:
            raise ValueError("notify.max_attempts must be a positive integer")
        proxy = data.get('proxy')
        if proxy is not None and (not isinstance(proxy, str) or
                                  not proxy.startswith(('http://', 'https://'))):
            raise ValueError('notify.proxy must be an HTTP proxy URL or null')
        if type(capacity) is not int or capacity < 1:
            raise ValueError("notify.queue_capacity must be a positive integer")
        if not enabled:
            return cls(False,
                       _resolve(root, helper) if helper else None,
                       str(chat_id) if chat_id else None, node, timeout, capacity, shutdown, proxy, attempts, retry)
        if not helper:
            raise ValueError("notify.enabled requires feishu_helper")
        helper_path = _resolve(root, helper)
        if not helper_path.is_file():
            raise ValueError(f"notify.feishu_helper does not exist: {helper_path}")
        if not chat_id or not str(chat_id).strip():
            raise ValueError("notify.enabled requires chat_id")
        return cls(True, helper_path, str(chat_id), node, timeout, capacity, shutdown, proxy, attempts, retry)


@dataclass(frozen=True)
class UnknownReviewConfig:
    continual_config: Path = ROOT / "configs/continual.json"
    maximum_sample_gap_seconds: float = 2.0
    maximum_visual_distance: float = 0.05
    geometry_tolerance: float = 0.02
    maximum_cases_per_session: int = 20
    maximum_candidates_per_frame: int = 16

    @classmethod
    def from_dict(cls, data, root: Path):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("unknown_review configuration must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        if set(data)-allowed:
            raise ValueError("unknown_review contains unsupported configuration")
        values = {"continual_config": _resolve(root, data.get("continual_config", "configs/continual.json"))}
        for name, default, upper in (("maximum_sample_gap_seconds", 2.0, 10.0),
                                     ("maximum_visual_distance", 0.05, 1.0),
                                     ("geometry_tolerance", 0.02, 0.2)):
            values[name] = _positive(f"unknown_review.{name}", data.get(name, default), upper=upper)
        for name, default in (("maximum_cases_per_session", 20), ("maximum_candidates_per_frame", 16)):
            value = data.get(name, default)
            if type(value) is not int or value < 1:
                raise ValueError(f"unknown_review.{name} must be a positive integer")
            values[name] = value
        return cls(**values)


@dataclass(frozen=True)
class LiveConfig:
    window_title_regex: str
    poll_seconds: float
    stable_frames: int
    layout_confirmations: int
    projection: str
    classifier: str
    cnn_checkpoint: Path
    tile_templates: Path
    glyph_templates: Path
    roi: Path
    vision: Path
    rules: Path
    menu: Path
    artifacts_root: Path
    match_end_silence_seconds: float
    persistent_issue_seconds: float
    decision: DecisionConfig
    execution: ExecutionConfig
    buttons: ButtonConfig
    button_templates: Path
    notify: NotifyConfig
    window_class_regex: str | None = None
    # False disables score OCR in the real-time pipeline: scores stay
    # unavailable (None) instead of blocking layout and state tracking.
    read_scores: bool = True
    lifecycle_ui: Path | None = None
    cnn_expected: dict = field(default_factory=dict)
    countdown: Path | None = None
    numeric_ocr: Path | None = None
    unknown_review: UnknownReviewConfig = field(default_factory=UnknownReviewConfig)
    riichi_selection: RiichiSelectionConfig = field(default_factory=RiichiSelectionConfig)
    riichi_detection: RiichiConfig = field(default_factory=RiichiConfig)
    emergency: EmergencyConfig = field(default_factory=EmergencyConfig)
    protection: ProtectionConfig = field(default_factory=ProtectionConfig)  # retired takeover supervisor
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    modal_manifest: Path | None = None  # reviewed in-game popups the Agent may dismiss

    @classmethod
    def load(cls, path: str | Path, *, root: Path = ROOT) -> "LiveConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Malformed live configuration")
        try:
            regex = str(data.get("window_title_regex", "雀魂"))
            re.compile(regex)
            class_regex = data.get("window_class_regex")
            if class_regex is not None:
                class_regex = str(class_regex)
                re.compile(class_regex)
            poll = _positive("poll_seconds", data.get("poll_seconds", 0.25),
                             upper=5.0)
            stable_frames = int(data.get("stable_frames", 2))
            if stable_frames < 1:
                raise ValueError("stable_frames must be at least one")
            confirmations = int(data.get("layout_confirmations", 2))
            if confirmations < 2:
                raise ValueError("layout_confirmations must be at least two")
            projection = str(data.get("projection", "auto"))
            if projection != "auto" and projection not in PROJECTION_NAMES:
                raise ValueError(
                    "projection must be auto or one of: " + ", ".join(PROJECTION_NAMES))
            classifier = str(data.get("classifier", "auto"))
            expected = data.get("cnn_expected", {})
            if not isinstance(expected, dict) or set(expected)-{'model_name','experiment','seed','sha256'}:
                raise ValueError("cnn_expected contains unsupported checkpoint metadata")
            if classifier not in CLASSIFIER_KINDS:
                raise ValueError(
                    f"classifier must be one of: {', '.join(CLASSIFIER_KINDS)}")
            read_scores = data.get("read_scores", True)
            if type(read_scores) is not bool:
                raise ValueError("read_scores must be boolean")
            silence = _positive("match_end_silence_seconds",
                                data.get("match_end_silence_seconds", 90))
            persistent = _positive("persistent_issue_seconds",
                                   data.get("persistent_issue_seconds", 10))
            buttons_data = data.get("buttons")
            if buttons_data is not None and not isinstance(buttons_data, dict):
                raise ValueError("buttons configuration must be a mapping")
            buttons_data = dict(buttons_data or {})
            button_templates = buttons_data.pop(
                "templates", "artifacts/templates/buttons")
            paths = {
                key: _resolve(root, data.get(key, default))
                for key, default in (
                    ("cnn_checkpoint",
                     "artifacts/model-select-20260929/convnext_tiny_best.pt"),
                    ("tile_templates", "artifacts/templates/tiles"),
                    ("glyph_templates", "artifacts/templates/glyphs"),
                    ("roi", "configs/roi.yaml"),
                    ("vision", "configs/vision.yaml"),
                    ("rules", "configs/sanma.json"),
                    ("menu", "configs/menu.steam.yaml"),
                    ("artifacts_root", "artifacts/live"),
                )
            }
            return cls(
                window_title_regex=regex,
                window_class_regex=class_regex,
                poll_seconds=poll,
                stable_frames=stable_frames,
                layout_confirmations=confirmations,
                projection=projection,
                classifier=classifier,
                **paths,
                match_end_silence_seconds=silence,
                persistent_issue_seconds=persistent,
                decision=DecisionConfig.from_dict(data.get("decision")),
                execution=ExecutionConfig.from_dict(data.get("execution")),
                buttons=ButtonConfig.from_dict(buttons_data),
                button_templates=_resolve(root, button_templates),
                notify=NotifyConfig.from_dict(data.get("notify"), root),
                read_scores=read_scores,
              lifecycle_ui=(_resolve(root, data["lifecycle_ui"])
                                if data.get("lifecycle_ui") else None),
              cnn_expected=expected,
              countdown=_resolve(root, data["countdown"]) if data.get("countdown") else None,
                numeric_ocr=_resolve(root, data['numeric_ocr']) if data.get('numeric_ocr') else None,
                unknown_review=UnknownReviewConfig.from_dict(data.get("unknown_review"), root),
                riichi_selection=RiichiSelectionConfig.from_dict(data.get("riichi_selection")),
                riichi_detection=RiichiConfig.from_dict(data.get("riichi_detection")),
                emergency=EmergencyConfig.from_dict(data.get("emergency")),
                protection=ProtectionConfig.from_dict(data.get('protection'),root),
                watchdog=WatchdogConfig.from_dict(data.get('watchdog')),
                modal_manifest=(_resolve(root, data['modal_manifest'])
                                if data.get('modal_manifest') else None),
            )
        except (TypeError, AttributeError) as exc:
            raise ValueError("Malformed live configuration") from exc
        except re.error as exc:
            raise ValueError("window_title_regex/window_class_regex must be valid regexes") from exc
