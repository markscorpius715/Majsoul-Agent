from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class ReviewKind(StrEnum):
    TILE = "tile"
    ACTION = "action"
    MELD = "meld"
    UI_STATE = "ui_state"
    METADATA = "metadata"


class UnknownReason(StrEnum):
    MODEL_UNCERTAINTY = "model_uncertainty"
    STATE_INCONSISTENCY = "state_inconsistency"
    NOVEL_VISUAL_PATTERN = "novel_visual_pattern"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


class PerceptionLoopState(StrEnum):
    """Control-plane states; these are not Mahjong GameEvent values."""

    WAIT_GAME = "WAIT_GAME"
    OBSERVE = "OBSERVE"
    STATE_UPDATE = "STATE_UPDATE"
    RESYNC = "RESYNC"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    SAFE_PAUSE = "SAFE_PAUSE"


@dataclass(frozen=True)
class Candidate:
    label: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "score": self.score}


@dataclass(eq=False)
class UnknownEvidence:
    kind: ReviewKind
    region: str
    timestamp: float
    frame_index: int
    reasons: tuple[UnknownReason, ...]
    full_frame: np.ndarray = field(repr=False)
    crop: np.ndarray = field(repr=False)
    previous_frame: np.ndarray | None = field(default=None, repr=False)
    next_frame: np.ndarray | None = field(default=None, repr=False)
    candidates: tuple[Candidate, ...] = ()
    source: str = "runtime_unknown"
    model_version: str | None = None
    state: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnknownAssessment:
    reasons: tuple[UnknownReason, ...]

    @property
    def unknown(self) -> bool:
        return bool(self.reasons)


@dataclass(frozen=True)
class PersistenceResult:
    persistent: bool
    new_case: bool
    safe_pause: bool
    streak: int
    fingerprint: str
    control_state: PerceptionLoopState


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    path: str
    status: ReviewStatus
    message: str
    created: bool = True
