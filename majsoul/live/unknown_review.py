"""Stable unknown UI evidence enters human review; it never trains a model."""
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np

from ..continual.config import ContinualConfig
from ..continual.detector import PersistentUnknownDetector, visual_embedding
from ..continual.schemas import Candidate, ReviewKind, UnknownEvidence, UnknownReason
from ..continual.store import ReviewCaseStore
from ..domain import Box
from .buttons import button_descriptor, _cosine_distance


@dataclass(eq=False)
class _Track:
    key: str
    kind: ReviewKind
    box: Box
    descriptor: np.ndarray
    samples: list = field(default_factory=list)
    completed: bool = False


class UnknownButtonReview:
    def __init__(self, config, directory, dataset_root, *, session, versions=None):
        self.config = config
        self.continual = ContinualConfig.load(config.continual_config)
        self.persistence = PersistentUnknownDetector(self.continual)
        self.store = ReviewCaseStore(directory, dataset_root)
        self.session = session
        self.versions = versions or {}
        self._tracks = []
        self._next_track = 0
        self._context = None
        self._previous = None
        self._previous_image = None
        self.created = 0

    def reset(self):
        self.persistence.end_frame(())
        self._tracks = []
        self._context = self._previous = self._previous_image = None

    def _descriptor(self, image, kind):
        return visual_embedding(image) if kind == ReviewKind.UI_STATE else button_descriptor(image, (48, 48))

    def _equivalent(self, old, descriptor, kind):
        return _cosine_distance(self._descriptor(old, kind), descriptor) <= self.config.maximum_visual_distance

    def observe(self, frame, index, observation, phase, unknowns, *, eligible,
                predictions=None, epoch=0):
        """Count each fresh stable frame once; gaps and invalid views reset it."""
        if (not eligible or not unknowns or phase not in ("discard", "response", "riichi_select") or
                observation.frame_index != index or observation.timestamp != frame.captured_at):
            self.reset()
            return []
        context = (observation.fingerprint, phase, tuple(frame.window), epoch)
        if self._previous is not None and (index <= self._previous[0] or frame.captured_at <= self._previous[1]):
            return []
        if (context != self._context or self._previous is not None and
                (index != self._previous[0]+1 or
                 not 0 < frame.captured_at-self._previous[1] <= self.config.maximum_sample_gap_seconds)):
            self.reset()
        self._context = context
        predictions = predictions or {}
        active, tracks, events = set(), [], []
        width, height = frame.image.shape[1], frame.image.shape[0]
        if len(unknowns) > self.config.maximum_candidates_per_frame:
            self.reset()
            return [{"state": "RESYNC", "reason": "too_many_unknown_candidates"}]
        for hint, crop, box in sorted(unknowns, key=lambda item: (item[2].x1, item[2].y1)):
            details = predictions.get(box, {})
            kind = ReviewKind.UI_STATE if details.get("clipped") else ReviewKind.ACTION
            descriptor = self._descriptor(crop, kind)
            if not np.any(descriptor):
                continue  # Uniform animation/background has no glyph evidence.
            if any(abs(box.center[0]-t.box.center[0])/width <= self.config.geometry_tolerance and
                   abs(box.center[1]-t.box.center[1])/height <= self.config.geometry_tolerance for t in tracks):
                continue  # Repeated callbacks in one frame cannot raise a streak.
            track = next((t for t in self._tracks if t not in tracks and t.kind == kind and
                max(abs(box.x1-t.box.x1)/width, abs(box.x2-t.box.x2)/width,
                    abs(box.y1-t.box.y1)/height, abs(box.y2-t.box.y2)/height) <= self.config.geometry_tolerance and
                _cosine_distance(t.descriptor, descriptor) <= self.config.maximum_visual_distance), None)
            if track is None:
                self._next_track += 1
                prefix = 'action_buttons_clipped' if kind == ReviewKind.UI_STATE else 'action_buttons'
                track = _Track(f"{prefix}[{self._next_track}]", kind, box, descriptor)
            tracks.append(track)
            active.add((kind.value, track.key))
            if track.completed:
                continue
            sample = {"frame_index": index, "timestamp": frame.captured_at,
                      "box": [box.x1, box.y1, box.x2, box.y2],
                      "crop_pixels_sha256": sha256(crop.tobytes()).hexdigest()}
            track.samples.append(sample)
            track.samples = track.samples[-self.continual.persistent_frames:]
            ranked = details.get("ranked_distances", ())
            evidence = UnknownEvidence(
                kind=kind, region=track.key, timestamp=frame.captured_at, frame_index=index,
                reasons=(UnknownReason.STATE_INCONSISTENCY if kind == ReviewKind.UI_STATE
                         else UnknownReason.MODEL_UNCERTAINTY,), full_frame=frame.image, crop=crop,
                previous_frame=self._previous_image,
                candidates=tuple(Candidate(label, 1-distance/2) for label, distance in ranked[:5]),
                source="live_action_button", model_version="button_template_cosine_v1",
                state=observation.to_dict(), metadata={
                    "player_count": len(observation.active_slots), "mode": "sanma" if len(observation.active_slots)==3 else "yonma",
                    "session_id": self.session, "epoch": epoch, "phase": phase,
                    "active_slots": list(observation.active_slots), "window": list(frame.window),
                    "hint": hint, "hint_is_label": False, "candidate_score_kind": "1_minus_cosine_distance_over_2_not_probability",
                    "prediction": details, "samples": list(track.samples),
                    "required_stable_frames": self.continual.persistent_frames,
                    "versions": self.versions, "control_state": "HUMAN_REVIEW",
                    "classification_vs_geometry": ("ROI_boundary_clipping" if kind == ReviewKind.UI_STATE
                                                   else "unclipped_candidate_rejected_by_template_classifier"),
                    "next_frame_available": False,
                    "previous_frame_index": self._previous[0] if self._previous else None})
            result = self.persistence.observe(evidence)
            if result.streak == 1:
                track.samples = [sample]
                events.append({"state": "RESYNC", "region": track.key,
                               "reason": "first_unknown_reobserve_current_window"})
            if not result.persistent:
                continue
            if self.created >= self.config.maximum_cases_per_session:
                track.completed = True
                self.persistence.clear(kind.value, track.key, forget_reported=True)
                events.append({"state": "RESYNC", "region": track.key, "reason": "review_case_limit"})
                continue
            try:
                case = self.store.create(evidence, dedup_hamming_distance=self.continual.dedup_hamming_distance,
                                         dedup_match=lambda old, d=descriptor, k=kind: self._equivalent(old, d, k))
            except (OSError, ValueError) as exc:
                events.append({"state": "RESYNC", "region": track.key, "reason": "review_storage_failed", "error": str(exc)})
                continue  # Persistent next sample retries; not falsely marked done.
            self.created += int(case.created)
            track.completed = True
            self.persistence.clear(kind.value, track.key, forget_reported=True)
            events.append({"state": "HUMAN_REVIEW", "region": track.key, "case": case,
                           "hint": hint, "kind": kind.value, "streak": result.streak})
        self._tracks = tracks
        self.persistence.end_frame(active)
        self._previous = (index, frame.captured_at)
        self._previous_image = frame.image.copy()
        return events
