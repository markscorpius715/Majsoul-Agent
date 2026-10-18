from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import cv2
import numpy as np

from .config import ContinualConfig
from .schemas import (PersistenceResult, UnknownAssessment, UnknownEvidence,
                      UnknownReason, PerceptionLoopState)


def difference_hash(image: np.ndarray) -> int:
    """Return a 64-bit perceptual hash for review-case persistence/dedup."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("difference_hash requires a non-empty image")
    gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3 else image)
    normalized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = normalized[:, 1:] > normalized[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def visual_embedding(image: np.ndarray) -> np.ndarray:
    """Small generic embedding used only for novelty gating, not classification."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
        raise ValueError("visual_embedding requires a non-empty BGR image")
    normalized = cv2.resize(image, (48, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((48, 64), (16, 16), (8, 8), (8, 8), 9)
    gradient = hog.compute(gray).reshape(-1).astype(np.float32)
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).reshape(-1)
    vector = np.concatenate((gradient, hist.astype(np.float32)))
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


class KnownVisualIndex:
    def __init__(self, embeddings=()):
        self._embeddings = [np.asarray(item, dtype=np.float32) for item in embeddings]

    def add(self, image: np.ndarray) -> None:
        self._embeddings.append(visual_embedding(image))

    def nearest_distance(self, image: np.ndarray) -> float | None:
        if not self._embeddings:
            return None
        target = visual_embedding(image)
        return min(max(0.0, min(2.0, 1.0 - float(np.dot(target, item))))
                   for item in self._embeddings)


def assess_unknown(config: ContinualConfig, *, confidence: float | None,
                   margin: float | None, state_issues=(),
                   novelty_distance: float | None = None) -> UnknownAssessment:
    reasons = []
    if (confidence is None or confidence < config.minimum_confidence or
            margin is None or margin < config.minimum_margin):
        reasons.append(UnknownReason.MODEL_UNCERTAINTY)
    if tuple(state_issues):
        reasons.append(UnknownReason.STATE_INCONSISTENCY)
    if (novelty_distance is not None and
            novelty_distance > config.novelty_distance):
        reasons.append(UnknownReason.NOVEL_VISUAL_PATTERN)
    return UnknownAssessment(tuple(reasons))


@dataclass
class _Pending:
    visual_hash: int
    streak: int


class PersistentUnknownDetector:
    """Require stable repeated unknowns and suppress duplicate notifications."""

    def __init__(self, config: ContinualConfig):
        self.config = config
        self._pending: dict[tuple[str, str], _Pending] = {}
        self._reported: dict[tuple[str, str], list[int]] = defaultdict(list)

    def observe(self, evidence: UnknownEvidence) -> PersistenceResult:
        key = (evidence.kind.value, evidence.region)
        visual_hash = difference_hash(evidence.crop)
        pending = self._pending.get(key)
        if (pending is not None and hamming_distance(
                pending.visual_hash, visual_hash) <= self.config.dedup_hamming_distance):
            pending.visual_hash = visual_hash
            pending.streak += 1
        else:
            pending = _Pending(visual_hash, 1)
            self._pending[key] = pending
        digest = sha256(
            f"{key[0]}:{key[1]}:{visual_hash:016x}".encode("utf-8")).hexdigest()
        if pending.streak < self.config.persistent_frames:
            return PersistenceResult(
                False, False, False, pending.streak, digest,
                PerceptionLoopState.RESYNC)
        duplicate = any(hamming_distance(old, visual_hash) <=
                        self.config.dedup_hamming_distance
                        for old in self._reported[key])
        if not duplicate:
            self._reported[key].append(visual_hash)
        return PersistenceResult(
            True, not duplicate, True, pending.streak, digest,
            PerceptionLoopState.HUMAN_REVIEW)

    def clear(self, kind: str, region: str, *, forget_reported: bool = False) -> None:
        self._pending.pop((kind, region), None)
        if forget_reported:
            self._reported.pop((kind, region), None)

    def end_frame(self, active_unknowns: Iterable[tuple[str, str]]) -> None:
        """Drop streaks that were not unknown in the current observed frame."""
        active = set(active_unknowns)
        for key in tuple(self._pending):
            if key not in active:
                self._pending.pop(key, None)
