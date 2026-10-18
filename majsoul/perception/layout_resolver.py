"""Conservative match-level seat-layout resolution from table metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calibration.roi_config import ROIConfig
from ..calibration.seat_layout import SeatLayoutTracker
from .metadata_reader import GameMetadataReader
from .schemas import GameMetadataReading


@dataclass(frozen=True)
class LayoutCandidateEvaluation:
    layout: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class SeatLayoutResolution:
    evaluations: tuple[LayoutCandidateEvaluation, ...]
    pending_layout: tuple[str, ...] | None
    pending_count: int
    locked_layout: tuple[str, ...] | None
    issues: tuple[str, ...] = ()

    @property
    def candidates(self) -> tuple[tuple[str, ...], ...]:
        return tuple(item.layout for item in self.evaluations if item.accepted)


class SeatLayoutResolver:
    """Resolve one configured layout only from complete, self-consistent metadata.

    A candidate is evidence, not a lock. The underlying tracker requires the same
    unique candidate in consecutive observations. Missing or ambiguous evidence
    breaks that streak so animation frames cannot bridge two unrelated readings.
    """

    def __init__(self, roi: ROIConfig, reader: GameMetadataReader, *,
                 starting_points: int, confirmations_required: int = 2):
        roi.validate()
        if type(starting_points) is not int or starting_points <= 0:
            raise ValueError("starting_points must be a positive integer")
        self.roi = roi
        self.reader = reader
        self.starting_points = starting_points
        self.tracker = SeatLayoutTracker(
            roi, confirmations_required=confirmations_required)

    def _candidate_issues(self, reading: GameMetadataReading,
                          expected_layout: tuple[str, ...]) -> tuple[str, ...]:
        issues = []
        if reading.active_slots != expected_layout:
            issues.append("Metadata reading does not match the evaluated layout")
        if not reading.complete:
            issues.append("Metadata reading is incomplete")
            if not isinstance(reading, GameMetadataReading):
                return tuple(issues)
            entries = [('round', reading.round), ('remaining_tiles', reading.remaining_tiles),
                       ('honba', reading.honba), ('riichi_sticks', reading.riichi_sticks)]
            entries.extend((f'seat_{slot}', seat) for slot, seat in reading.seats.items())
            if reading.scores_available:
                entries.extend((f'score_{slot}', score) for slot, score in reading.scores.items())
            issues.extend(f'{name}: {issue}' for name, field in entries for issue in field.issues)
            return tuple(issues)

        expected_winds = set(("1z", "2z", "3z", "4z")[:self.roi.player_count])
        winds = [seat.wind for seat in reading.seats.values()]
        if set(winds) != expected_winds or len(set(winds)) != len(winds):
            issues.append("Seat winds do not match the configured player count")
        if sum(seat.dealer is True for seat in reading.seats.values()) != 1:
            issues.append("Layout does not contain exactly one dealer")
        if (reading.round.hand_number is None or
                not 1 <= reading.round.hand_number <= self.roi.player_count):
            issues.append("Hand number is invalid for the configured player count")
        counters = (reading.remaining_tiles.value, reading.honba.value,
                    reading.riichi_sticks.value)
        if (counters[0] is None or not 0 <= counters[0] <= 200 or
                counters[1] is None or not 0 <= counters[1] <= 99 or
                counters[2] is None or not 0 <= counters[2] <= 99):
            issues.append("Round counters are outside supported ranges")
        scores = [score.value for score in reading.scores.values()]
        if not getattr(reading, "scores_available", True):
            pass  # Scores are unavailable by configuration; never fabricate them.
        elif any(score is None or not -999900 <= score <= 999900 or score % 100
                 for score in scores):
            issues.append("Scores must be bounded multiples of 100")
        elif sum(scores) + reading.riichi_sticks.value * 1000 != (
                self.roi.player_count * self.starting_points):
            issues.append("Scores plus riichi deposits do not conserve match points")
        return tuple(issues)

    def observe(self, image: np.ndarray) -> SeatLayoutResolution:
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or
                image.shape[2] != 3 or image.size == 0):
            raise ValueError("Seat layout resolution requires a non-empty BGR image")
        if self.tracker.locked_layout is not None:
            layout = self.tracker.locked_layout
            return SeatLayoutResolution(
                (LayoutCandidateEvaluation(layout, ()),), layout,
                self.tracker.pending_count, layout)

        evaluations = []
        for variant in self.roi.layout_variants:
            reading = self.reader.read(image, self.roi, variant)
            layout = tuple(variant)
            evaluations.append(LayoutCandidateEvaluation(
                layout, self._candidate_issues(reading, layout)))
        accepted = [item.layout for item in evaluations if item.accepted]
        issues = []
        if len(accepted) == 1:
            locked = self.tracker.observe(accepted[0])
        else:
            locked = self.tracker.observe_unknown()
            if not accepted:
                issues.append("No layout candidate has complete, consistent metadata")
            else:
                issues.append("Multiple layout candidates have equally valid metadata")
        return SeatLayoutResolution(
            tuple(evaluations), self.tracker.pending_layout,
            self.tracker.pending_count, locked, tuple(issues))

    def reset_match(self) -> None:
        self.tracker.reset_match()
