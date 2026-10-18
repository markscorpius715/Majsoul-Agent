from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2

from ..calibration.frames import read_image
from ..calibration.roi_config import ROIConfig
from ..perception import TileTemplateClassifier, VisibleTilesReader, VisionConfig
from ..perception.roi_extractor import extract_roi
from ..perception.schemas import MeldReading, TileSequenceReading
from ..state.schemas import FrameObservation
from .config import ContinualConfig
from .detector import KnownVisualIndex, PersistentUnknownDetector, assess_unknown
from .schemas import Candidate, ReviewKind, UnknownEvidence, UnknownReason
from .store import ReviewCaseStore


def _readings(visible):
    yield ReviewKind.TILE, visible.hand
    yield ReviewKind.TILE, visible.dora
    for reading in visible.rivers.values():
        yield ReviewKind.TILE, reading
    for reading in visible.melds.values():
        yield ReviewKind.MELD, reading
    for reading in visible.kita.values():
        yield ReviewKind.TILE, reading


def _recognized(reading):
    if isinstance(reading, MeldReading):
        return [item for group in reading.groups for item in group]
    return list(reading.tiles)


def _labels(reading):
    if isinstance(reading, MeldReading):
        return [list(group) for group in reading.labels]
    return list(reading.labels)


class UnknownVideoCollector:
    """Collect persistent visual unknowns from offline video without training."""

    def __init__(self, roi: ROIConfig, vision: VisionConfig,
                 classifier: TileTemplateClassifier, active_slots,
                 config: ContinualConfig, store: ReviewCaseStore,
                 known_index: KnownVisualIndex, *, source: str = "runtime_unknown"):
        self.roi = roi
        self.reader = VisibleTilesReader(vision, classifier)
        self.active_slots = tuple(active_slots)
        self.config = config
        self.store = store
        self.known_index = known_index
        self.persistence = PersistentUnknownDetector(config)
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty identifier")
        self.source = source.strip()
        self.source_context = {
            "player_count": roi.player_count,
            "mode": "yonma" if roi.player_count == 4 else "sanma",
        }

    def observe(self, image, *, timestamp: float, frame_index: int,
                previous_frame=None, state=None,
                observation: FrameObservation | None = None,
                state_issues=()) -> list[UnknownEvidence]:
        visible = self.reader.read(image, self.roi, self.active_slots)
        pending = []
        active_unknowns: set[tuple[str, str]] = set()
        for kind, reading in _readings(visible):
            context = {**self.source_context,
                       "labels": _labels(reading), "issues": list(reading.issues),
                       "active_slots": list(visible.active_slots)}
            for index, item in enumerate(_recognized(reading)):
                prediction = item.prediction
                novelty = self.known_index.nearest_distance(item.detection.image)
                assessment = assess_unknown(
                    self.config, confidence=prediction.confidence,
                    margin=prediction.margin, novelty_distance=novelty)
                if not assessment.unknown:
                    continue
                evidence = UnknownEvidence(
                    kind=kind,
                    region=f"{reading.region}[{index}]",
                    timestamp=timestamp,
                    frame_index=frame_index,
                    reasons=assessment.reasons,
                    full_frame=image.copy(),
                    crop=item.detection.image.copy(),
                    previous_frame=(previous_frame.copy()
                                    if previous_frame is not None else None),
                    candidates=tuple(Candidate(label, score)
                                     for label, score in prediction.candidates),
                    source=self.source,
                    model_version="tile_template_v0",
                    state=state,
                    metadata={**context, "distance": prediction.distance,
                              "margin": prediction.margin,
                              "orientation": prediction.orientation},
                )
                active_unknowns.add((evidence.kind.value, evidence.region))
                if self.persistence.observe(evidence).new_case:
                    pending.append(evidence)
            structural_issues = [issue for issue in reading.issues
                                 if not issue.startswith("Unknown ")]
            if structural_issues:
                region = reading.region
                crop = extract_roi(image, self.roi, region).image
                evidence = UnknownEvidence(
                    kind=kind,
                    region=region,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    reasons=(UnknownReason.STATE_INCONSISTENCY,),
                    full_frame=image.copy(),
                    crop=crop.copy(),
                    previous_frame=(previous_frame.copy()
                                    if previous_frame is not None else None),
                    source=self.source,
                    model_version="tile_template_v0",
                    state=state,
                    metadata={**context, "issues": structural_issues},
                )
                active_unknowns.add((evidence.kind.value, evidence.region))
                if self.persistence.observe(evidence).new_case:
                    pending.append(evidence)
        if observation is not None:
            grouped_metadata_issues: dict[str, list[str]] = {}
            for issue in observation.issues:
                prefix = issue.split(":", 1)[0]
                region = ("round_info" if prefix == "round" else prefix)
                if (region in {"remaining_tiles", "honba", "riichi_sticks"} or
                        region.startswith("score_") or region.startswith("seat_")):
                    grouped_metadata_issues.setdefault(region, []).append(issue)
            for region, issues in grouped_metadata_issues.items():
                crop = extract_roi(image, self.roi, region).image
                evidence = UnknownEvidence(
                    kind=ReviewKind.METADATA,
                    region=region,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    reasons=(UnknownReason.MODEL_UNCERTAINTY,),
                    full_frame=image.copy(),
                    crop=crop.copy(),
                    previous_frame=(previous_frame.copy()
                                    if previous_frame is not None else None),
                    source=self.source,
                    model_version="glyph_template_v0",
                    state=state,
                    metadata={**self.source_context, "issues": issues,
                              "active_slots": list(observation.active_slots)},
                )
                active_unknowns.add((evidence.kind.value, evidence.region))
                if self.persistence.observe(evidence).new_case:
                    pending.append(evidence)
            validation_issues = tuple(state_issues)
            if observation.complete and validation_issues:
                evidence = UnknownEvidence(
                    kind=ReviewKind.UI_STATE,
                    region="table_state",
                    timestamp=timestamp,
                    frame_index=frame_index,
                    reasons=(UnknownReason.STATE_INCONSISTENCY,),
                    full_frame=image.copy(),
                    crop=image.copy(),
                    previous_frame=(previous_frame.copy()
                                    if previous_frame is not None else None),
                    source=self.source,
                    model_version="state_validator_v0",
                    state=state,
                    metadata={**self.source_context,
                              "issues": list(validation_issues),
                              "observation": observation.to_dict()},
                )
                active_unknowns.add((evidence.kind.value, evidence.region))
                if self.persistence.observe(evidence).new_case:
                    pending.append(evidence)
        self.persistence.end_frame(active_unknowns)
        return pending

    def finalize(self, evidence: UnknownEvidence, next_frame):
        return self.store.create(
            replace(evidence, next_frame=next_frame.copy()),
            dedup_hamming_distance=self.config.dedup_hamming_distance)


def build_known_index(template_root: str | Path) -> KnownVisualIndex:
    index = KnownVisualIndex()
    for path in sorted(Path(template_root).glob("*/*")):
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            index.add(read_image(path))
    return index


def iter_video(path: str | Path, *, start: int = 0, end: int | None = None,
               stride: int = 1):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Unable to open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        stop = total if end is None else min(end, total)
        if fps <= 0 or start < 0 or start >= stop or stride < 1:
            raise ValueError("Invalid video metadata, frame range, or stride")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_index in range(start, stop):
            if (frame_index - start) % stride == 0:
                ok, image = capture.read()
                if not ok:
                    raise OSError(f"Failed to decode frame {frame_index}")
                yield image, frame_index, frame_index / fps
            elif not capture.grab():
                raise OSError(f"Failed to decode frame {frame_index}")
    finally:
        capture.release()
