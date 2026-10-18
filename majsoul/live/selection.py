"""Client selection evidence, bound to the input that opened the window."""
from dataclasses import dataclass
import math

import cv2
import numpy as np
from .selection_marker import SelectionCancelMarker
from .gates import input_issues


@dataclass(frozen=True)
class RiichiSelectionConfig:
    bright_minimum: float = 170.0
    enabled_ratio: tuple[float, float] = (.90, 1.10)
    disabled_ratio: tuple[float, float] = (.45, .75)
    brightness_percentile: float = 90.0
    target_tolerance: float = .02
    cancel_marker_manifest: str | None = None
    cancel_marker_minimum_score: float = .90
    cancel_marker_minimum_margin: float = .05
    cancel_marker_regions: tuple[str, ...] = ('countdown',)

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return cls()
        if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError('Invalid riichi_selection configuration')
        values = dict(data)
        try:
            for key in ('enabled_ratio', 'disabled_ratio', 'cancel_marker_regions'):
                if key in values:
                    if isinstance(values[key],(str,bytes)):
                        raise ValueError('Selection arrays must not be strings')
                    values[key] = tuple(values[key])
            config = cls(**values)
            numeric = (config.bright_minimum, config.brightness_percentile,
                       config.target_tolerance, *config.enabled_ratio, *config.disabled_ratio,
                       config.cancel_marker_minimum_score, config.cancel_marker_minimum_margin)
            if any(isinstance(x, bool) or not isinstance(x, (float, int)) for x in numeric):
                raise ValueError('Non-numeric riichi_selection threshold')
        except (TypeError, AttributeError) as exc:
            raise ValueError('Invalid riichi_selection configuration') from exc
        if (not math.isfinite(config.bright_minimum) or not 0 < config.bright_minimum <= 255 or
                not math.isfinite(config.brightness_percentile) or not 50 <= config.brightness_percentile < 100 or
                not math.isfinite(config.target_tolerance) or not 0 < config.target_tolerance <= .2 or
                any(len(pair) != 2 or any(not math.isfinite(x) for x in pair) or
                    not 0 < pair[0] < pair[1] <= 2
                    for pair in (config.enabled_ratio, config.disabled_ratio)) or
                config.disabled_ratio[1] >= config.enabled_ratio[0] or
                not 0 < config.cancel_marker_minimum_score <= 1 or
                not 0 < config.cancel_marker_minimum_margin <= 1 or
                not config.cancel_marker_regions or
                any(not isinstance(n,str) or not n.strip() for n in config.cancel_marker_regions) or
                config.cancel_marker_manifest is not None and
                (not isinstance(config.cancel_marker_manifest,str) or not config.cancel_marker_manifest.strip())):
            raise ValueError('Invalid riichi_selection thresholds')
        return config


def hand_brightness(image, hand, percentile):
    values = []
    height, width = image.shape[:2]
    for tile in hand.tiles:
        box = tile.detection.box
        x1, y1, x2, y2 = (round(v) for v in (box.x1, box.y1, box.x2, box.y2))
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError('Selection hand target is clipped')
        crop = image[y1:y2, x1:x2]
        values.append(float(np.percentile(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2], percentile)))
    return tuple(values)


class RiichiSelection:
    """A sent riichi input plus a recognized bright/dim partition proves selection.

    A missing button, elapsed time, a guessed tenpai set, or an unchanged white
    hand alone cannot open this window. One complete mask suffices; public
    animation cannot invalidate the bound local selection evidence.
    """
    def __init__(self, config):
        self.config = config
        self.cancel_marker = SelectionCancelMarker(config)
        self.reset()

    def reset(self):
        self.before = self.window = self.values = self.boxes = None
        self.deadline = None
        self.discard_sent = False
        self.cancel_before = None

    def begin(self, frame, observation, visible, deadline, *, roi=None):
        if ('riichi' not in (observation.action_buttons or ()) or
                input_issues(observation) or not visible.hand.complete or
                len(visible.hand.tiles) != len(observation.hand) or not observation.hand or
                observation.timestamp != frame.captured_at or
                not math.isfinite(deadline) or
                deadline <= frame.captured_at or
                tuple(t.prediction.tile for t in visible.hand.tiles) !=
                tuple(t.tile for t in observation.hand)):
            raise ValueError('Riichi input lacks a usable current baseline')
        values = hand_brightness(frame.image, visible.hand, self.config.brightness_percentile)
        if any(value < self.config.bright_minimum for value in values):
            raise ValueError('Riichi baseline hand is already dim')
        marker = self.cancel_marker.read(frame,roi)
        if marker['accepted'] or marker.get('possible'):
            raise ValueError('Riichi baseline already shows a selection cancel marker')
        self.before, self.window, self.values = observation, tuple(frame.window), values
        self.boxes = tuple(tile.detection.box for tile in visible.hand.tiles)
        self.deadline, self.discard_sent = deadline, False
        self.cancel_before = marker if roi is not None and self.cancel_marker.template is not None else None

    def read(self, frame, observation, visible, *, roi=None):
        def unknown(reason, **evidence):
            return None, {'status': 'unknown', 'reason': reason, **evidence}

        before = self.before
        if before is None:
            return unknown('no_sent_riichi_input')
        if (tuple(frame.window) != self.window or any(
                getattr(observation, name) is not None and getattr(before, name) is not None and
                getattr(observation, name) != getattr(before, name)
                for name in ('round_wind', 'hand_number', 'honba')) or
                observation.active_slots != before.active_slots):
            return unknown('selection_window_changed')
        if (observation.timestamp != frame.captured_at or
                observation.timestamp <= before.timestamp or
                observation.frame_index <= before.frame_index or observation.timestamp >= self.deadline):
            return unknown('selection_observation_outside_budget')
        if (input_issues(observation) or not visible.hand.complete or
                tuple(t.tile for t in observation.hand) != tuple(t.tile for t in before.hand) or
                len(visible.hand.tiles) != len(before.hand) or
                tuple(t.prediction.tile for t in visible.hand.tiles) != tuple(t.tile for t in before.hand)):
            return unknown('selection_table_or_hand_changed')
        if 'riichi' in (observation.action_buttons or ()):
            return unknown('riichi_button_still_visible')
        width, height = frame.window[2:]
        for old, tile in zip(self.boxes, visible.hand.tiles):
            new = tile.detection.box
            if max(abs(old.x1-new.x1)/width, abs(old.x2-new.x2)/width,
                   abs(old.y1-new.y1)/height, abs(old.y2-new.y2)/height) > self.config.target_tolerance:
                return unknown('selection_hand_geometry_changed')
        values = hand_brightness(frame.image, visible.hand, self.config.brightness_percentile)
        ratios = [after / old for old, after in zip(self.values, values)]
        enabled, disabled = [], []
        for index, ratio in enumerate(ratios):
            if self.config.enabled_ratio[0] <= ratio <= self.config.enabled_ratio[1]:
                enabled.append(index)
            elif self.config.disabled_ratio[0] <= ratio <= self.config.disabled_ratio[1]:
                disabled.append(index)
            else:
                return unknown('selection_brightness_ambiguous', ratios=ratios)
        marker = self.cancel_marker.read(frame,roi) if not disabled else None
        if not enabled or (not disabled and not (self.cancel_before is not None and
                                                not self.cancel_before['accepted'] and marker['accepted'])):
            return unknown('selection_partition_unproved', ratios=ratios)
        return tuple(enabled), {'status': 'complete', 'source': ('client_bright_dim_partition' if disabled
                                else 'client_cancel_marker_and_undimmed_hand'),
                                'before_frame_index': before.frame_index,
                                'enabled_indices': enabled, 'disabled_indices': disabled,
                                'brightness': values, 'ratios': ratios,
                                'cancel_marker': marker,
                                'deadline': self.deadline}
