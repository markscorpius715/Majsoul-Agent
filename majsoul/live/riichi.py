"""Per-seat stick comparison against a match-opening, immutable empty baseline."""

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

from ..perception.roi_extractor import extract_roi


@dataclass(frozen=True)
class RiichiConfig:
    enabled: bool = False
    confirmations: int = 3
    minimum_difference: float = .12
    pixel_difference: float = .20
    changed_fraction: float = .25
    minimum_white_fraction: float = .22
    minimum_red_fraction: float = .002
    maximum_repeat_difference: float = .08
    minimum_sample_seconds: float = .08
    maximum_sample_gap: float = 3.0

    @classmethod
    def from_dict(cls, data):
        obj = cls(**(data or {}))
        if type(obj.enabled) is not bool or type(obj.confirmations) is not int or obj.confirmations < 2:
            raise ValueError('riichi_detection requires bool enabled and confirmations >= 2')
        for key, value in vars(obj).items():
            if key in ('enabled', 'confirmations'):
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid riichi_detection ' + key)
            if not key.endswith(('seconds', 'gap')) and value > 1:
                raise ValueError('riichi_detection ratios must be <= 1')
        return obj


class RiichiDetector:
    def __init__(self, config):
        self.config = config
        self.baselines = {}
        self.geometry = None
        self.new_hand()

    def new_hand(self):
        # Baselines deliberately survive tracker and hand restarts.
        self.latched = set()
        self.samples = {}
        self.hand_id = None
        self.next_hand = None

    def _patch(self, image, roi, slot):
        patch = extract_roi(image, roi, 'riichi_' + slot).image
        return cv2.resize(patch, (80, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

    def apply(self, image, roi, observation):
        if not self.config.enabled:
            return observation
        hand_id = observation.round_id
        if hand_id is not None and not any(i.startswith(('round:', 'honba:')) for i in observation.issues):
            if self.hand_id is None:
                self.hand_id = hand_id
            elif hand_id == self.hand_id:
                self.next_hand = None
            elif (self.next_hand is not None and self.next_hand[0] == hand_id and
                  observation.timestamp > self.next_hand[1]):
                # Also works without the optional settlement UI controller.
                # A single bad OCR sample must not erase a proven declaration.
                # The first sample already belongs to the candidate new hand;
                # keep its ROI measurement, while clearing old declarations.
                first_samples = self.samples
                self.new_hand()
                self.samples = first_samples
                self.hand_id = hand_id
            else:
                self.next_hand = (hand_id, observation.timestamp)
                self.samples.clear()
        geometry = (image.shape[:2], tuple((slot, tuple(roi.regions.get('riichi_' + slot) or ()))
                                         for slot in observation.active_slots))
        # Never learn an in-progress hand/stick as an empty baseline. Zero deposits
        # alone is insufficient when joining late or after a settlement.
        opening = (observation.hand_number == 1 and observation.honba == 0 and
                   observation.riichi_sticks == 0 and observation.remaining_tiles is not None and
                   observation.remaining_tiles >= 53 and len(observation.hand) in (13, 14) and
                   all(t.tile is not None for t in observation.hand) and
                   all(not p.river and not p.melds and p.kita_count == 0 and p.seat is not None
                       for p in observation.players))
        players = []
        for player in observation.players:
            slot = player.slot
            evidence = {'source': 'match_opening_roi', 'status': 'unknown'}
            marker = None
            if roi.regions.get('riichi_' + slot) is not None:
                patch = self._patch(image, roi, slot)
                if opening and slot not in self.baselines and (self.geometry is None or self.geometry == geometry):
                    self.geometry = geometry
                    self.baselines[slot] = patch.copy()
                    evidence['baseline_frame'] = observation.frame_index
                baseline = self.baselines.get(slot)
                if baseline is not None and geometry == self.geometry:
                    delta = np.max(abs(patch - baseline), axis=2)
                    difference = float(np.mean(abs(patch - baseline)))
                    changed = float(np.mean(delta >= self.config.pixel_difference))
                    # A changed ROI is evidence only if it resembles the white
                    # stick with a red centre, not a turn highlight or an overlay.
                    hsv = cv2.cvtColor((patch * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
                    white = float(np.mean((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 185)))
                    red = float(np.mean(((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170)) &
                                        (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 100)))
                    positive = (difference >= self.config.minimum_difference and
                                changed >= self.config.changed_fraction and
                                white >= self.config.minimum_white_fraction and red >= self.config.minimum_red_fraction)
                    previous = self.samples.get(slot)
                    count = 0
                    if positive:
                        count = 1
                        if previous is not None:
                            timestamp, prior_patch, prior_count = previous
                            gap = observation.timestamp - timestamp
                            if gap < self.config.minimum_sample_seconds:
                                count = prior_count
                            elif gap <= self.config.maximum_sample_gap and float(np.mean(abs(patch - prior_patch))) <= self.config.maximum_repeat_difference:
                                count = prior_count + 1
                        if previous is None or observation.timestamp - previous[0] >= self.config.minimum_sample_seconds:
                            self.samples[slot] = (observation.timestamp, patch, count)
                        if count >= self.config.confirmations:
                            self.latched.add(slot)
                    else:
                        self.samples.pop(slot, None)
                    marker = True if slot in self.latched else False if difference < self.config.minimum_difference else None
                    evidence.update(status='confirmed' if marker is True else 'empty' if marker is False else 'retesting',
                                    difference=difference, changed_fraction=changed,
                                    white_fraction=white, red_fraction=red, confirmations=count)
                else:
                    evidence['reason'] = 'baseline_missing_or_geometry_changed'
            sideways = any(t.tile is not None and t.orientation.startswith('horizontal')
                           for t in player.river)
            # Perspective can make an ordinary second-row tile look sideways.
            # Keep this diagnostic only; it cannot bypass stick confirmation.
            evidence['sideways_candidate'] = sideways
            if slot in self.latched:
                marker = True
            players.append(replace(player, riichi=marker, riichi_evidence=evidence))
        return replace(observation, players=tuple(players))
