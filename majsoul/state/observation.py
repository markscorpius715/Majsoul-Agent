from __future__ import annotations

import numpy as np

from ..calibration.roi_config import ROIConfig
from ..perception import (GameMetadataReader, GlyphTemplateClassifier,
                          TileTemplateClassifier, VisibleTilesReader, VisionConfig)
from ..perception.schemas import RecognizedTile, VisibleTilesReading
from .schemas import (FrameObservation, ObservedMeld, ObservedTile,
                      PlayerObservation)


def _tile(item: RecognizedTile) -> ObservedTile:
    box = item.detection.box
    return ObservedTile(item.prediction.tile, item.prediction.orientation,
                        item.prediction.confidence, item.detection.face_down,
                        (box.x1, box.y1, box.x2, box.y2))


def _minimum_confidence(items) -> float:
    values = [item.confidence for item in items]
    return min(values) if values else 1.0


class FrameObservationReader:
    """Combine Phase 3 visible tiles and Phase 4 metadata into one snapshot."""

    # Class-level default keeps readers assembled without __init__ working.
    button_reader = None
    supports_local_input = False

    def read_local(self, image, roi, active_slots, *, timestamp, frame_index):
        """No metadata, rivers, melds or North-area dependency for the watchdog."""
        from ..perception.hand_reader import HandReader
        from ..perception.roi_extractor import extract_roi
        from types import SimpleNamespace
        hand = HandReader(self.visible_reader.config, self.visible_reader.classifier).read(
            extract_roi(image, roi, 'self_hand'))
        buttons = self.button_reader.read(image, roi) if self.button_reader is not None else None
        issues = tuple('hand: ' + issue for issue in hand.issues)
        if buttons is None:
            issues += ('action_buttons: reader unavailable',)
        # Unknown candidates must never be treated as an absent interaction.
        predictions = getattr(self.button_reader, 'last_predictions', {})
        if any(not p.get('accepted') for p in predictions.values()):
            issues += ('action_buttons: unknown candidates',)
        clock = self.countdown_reader.read(image, roi).to_dict() if self.countdown_reader is not None else None
        obs = FrameObservation(float(timestamp), frame_index, tuple(active_slots),
            None, None, None, None, None, (), tuple(_tile(t) for t in hand.tiles),
            hand.drawn_index, (), action_buttons=None if buttons is None else tuple(sorted(buttons)),
            issues=issues, countdown=clock)
        return obs, SimpleNamespace(hand=hand), buttons or {}

    def __init__(self, config: VisionConfig, tile_classifier: TileTemplateClassifier,
                 glyph_classifier: GlyphTemplateClassifier, button_reader=None, *,
                 read_scores: bool = True, countdown_reader=None, numeric_ocr=None):
        self.supports_local_input = True
        self.visible_reader = VisibleTilesReader(config, tile_classifier)
        self.metadata_reader = GameMetadataReader(
            config, glyph_classifier, read_scores=read_scores, numeric_ocr=numeric_ocr)
        self.button_reader = button_reader
        self.countdown_reader = countdown_reader

    def read(self, image: np.ndarray, roi: ROIConfig, active_slots, *,
             timestamp: float, frame_index: int) -> FrameObservation:
        observation, _ = self.read_with_visible(
            image, roi, active_slots, timestamp=timestamp, frame_index=frame_index)
        return observation

    def read_with_visible(self, image: np.ndarray, roi: ROIConfig, active_slots, *,
                          timestamp: float, frame_index: int
                          ) -> tuple[FrameObservation, VisibleTilesReading]:
        if timestamp < 0 or frame_index < 0:
            raise ValueError("timestamp and frame_index must be non-negative")
        action_buttons = None
        issues_extra = []
        occlusion_boxes = ()
        if self.button_reader is not None:
            try:
                action_buttons = tuple(sorted(self.button_reader.read(image, roi)))
            except Exception:
                issues_extra.append("action_buttons: reader failed")
            occlusion_boxes = getattr(self.button_reader, "last_occlusion_boxes", ())
        visible = (self.visible_reader.read(image, roi, active_slots,
                                           occlusion_boxes=occlusion_boxes)
                   if occlusion_boxes else self.visible_reader.read(image, roi, active_slots))
        metadata = self.metadata_reader.read(image, roi, active_slots)
        issues = [
            issue for issue in visible.issues
            if not (issue.startswith("melds_") and
                    ": Unknown meld tiles at positions:" in issue)
        ]
        if not metadata.round.complete:
            issues.extend(f"round: {issue}" for issue in metadata.round.issues)
        for name, reading in (("remaining_tiles", metadata.remaining_tiles),
                              ("honba", metadata.honba),
                              ("riichi_sticks", metadata.riichi_sticks)):
            issues.extend(f"{name}: {issue}" for issue in reading.issues)
        for slot, reading in metadata.scores.items():
            issues.extend(f"score_{slot}: {issue}" for issue in reading.issues)
        for slot, reading in metadata.seats.items():
            issues.extend(f"seat_{slot}: {issue}" for issue in reading.issues)
        issues.extend(metadata.issues)
        issues.extend(issues_extra)
        if visible.active_slots != metadata.active_slots:
            issues.append("Visible-tile and metadata layouts disagree")

        confidence = {
            "round": _minimum_confidence(metadata.round.glyphs),
            "remaining_tiles": _minimum_confidence(metadata.remaining_tiles.glyphs),
            "honba": _minimum_confidence(metadata.honba.glyphs),
            "riichi_sticks": _minimum_confidence(metadata.riichi_sticks.glyphs),
            "hand": _minimum_confidence(
                item.prediction for item in visible.hand.tiles),
            "dora_indicators": _minimum_confidence(
                item.prediction for item in visible.dora.tiles),
        }
        for slot in metadata.active_slots:
            confidence[f"score_{slot}"] = _minimum_confidence(
                metadata.scores[slot].glyphs)
            seat = metadata.seats[slot]
            confidence[f"seat_{slot}"] = (
                seat.glyph.confidence if seat.glyph is not None else 0.0)
            confidence[f"river_{slot}"] = _minimum_confidence(
                item.prediction for item in visible.rivers[slot].tiles)
            confidence[f"melds_{slot}"] = _minimum_confidence(
                item.prediction for group in visible.melds[slot].groups for item in group)
            kita = visible.kita.get(slot)
            if kita is not None:
                confidence[f"kita_{slot}"] = _minimum_confidence(
                    item.prediction for item in kita.tiles)
        unavailable_regions = tuple(
            issue.split(":", 1)[0] for issue in visible.issues
            if ": UI occlusion; region unavailable" in issue)
        for region in unavailable_regions:
            confidence["hand" if region == "hand" else
                       "dora_indicators" if region == "dora" else region] = 0.0
        confidence["overall"] = min(confidence.values())

        players = []
        for slot in metadata.active_slots:
            score = metadata.scores[slot]
            seat = metadata.seats[slot]
            river = visible.rivers[slot]
            melds = visible.melds[slot]
            kita = visible.kita.get(slot)
            players.append(PlayerObservation(
                slot=slot,
                seat=seat.wind,
                dealer=seat.dealer,
                score=score.value,
                river=tuple(_tile(item) for item in river.tiles),
                melds=tuple(ObservedMeld(
                    tuple(_tile(item) for item in group),
                    melds.kinds[index] if melds.kinds else "unknown",
                    melds.opened[index] if melds.opened else None)
                    for index, group in enumerate(melds.groups)),
                kita_count=(len(kita.tiles) if kita is not None and kita.complete
                            else 0 if not roi.kita else None),
                score_available=getattr(metadata, "scores_available", True),
            ))
        observation = FrameObservation(
            timestamp=float(timestamp),
            frame_index=int(frame_index),
            active_slots=metadata.active_slots,
            round_wind=metadata.round.round_wind,
            hand_number=metadata.round.hand_number,
            honba=metadata.honba.value,
            riichi_sticks=metadata.riichi_sticks.value,
            remaining_tiles=metadata.remaining_tiles.value,
            players=tuple(players),
            hand=tuple(_tile(item) for item in visible.hand.tiles),
            drawn_index=visible.hand.drawn_index,
            dora_indicators=tuple(_tile(item) for item in visible.dora.tiles),
            action_buttons=action_buttons,
            turn_indicator=None,
            issues=tuple(issues),
            confidence=confidence,
            countdown=(self.countdown_reader.read(image, roi).to_dict()
                       if getattr(self, "countdown_reader", None) is not None else None),
            ui_occlusions=tuple((box.x1, box.y1, box.x2, box.y2)
                                for box in occlusion_boxes),
            unavailable_regions=unavailable_regions,
        )
        return observation, visible
