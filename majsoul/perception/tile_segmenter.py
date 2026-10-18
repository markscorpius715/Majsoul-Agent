from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np

from ..domain import Box
from .config import ComponentLimits, VisionConfig
from .roi_extractor import ROICrop
from .schemas import SegmentationResult, SegmentedTile


@dataclass(frozen=True)
class _Component:
    x: int
    y: int
    width: int
    height: int
    area: int

    @property
    def center(self):
        return self.x + self.width / 2, self.y + self.height / 2


def _face_components(crop: np.ndarray, config: VisionConfig):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, config.value_min),
                       (179, config.saturation_max, 255))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return mask, [_Component(*(int(v) for v in stat)) for stat in stats[1:]]


def _filter_components(components: list[_Component], size: tuple[int, int],
                       limits: ComponentLimits, minimum_fill_ratio: float):
    width, height = size
    min_width, max_width = (value * width for value in limits.width_fraction)
    min_height, max_height = (value * height for value in limits.height_fraction)
    accepted = [item for item in components
                if min_width <= item.width <= max_width
                and min_height <= item.height <= max_height
                and item.area / (item.width * item.height) >= minimum_fill_ratio]
    oversized = [item for item in components
                 if item.width > max_width and item.height >= min_height
                 and item.area / (item.width * item.height) >= minimum_fill_ratio]
    return accepted, oversized


def _outside_ui_components(components, crop, excluded_boxes):
    # Remove proven UI components before row grouping, merging or anomaly
    # checks. Filtering final tiles is too late: UI glyphs can already have
    # changed the river grid or raised an oversized/off-grid issue.
    if not excluded_boxes:
        return components
    result = []
    for item in components:
        box = crop.map_box_to_screen(Box(item.x, item.y,
                                        item.x + item.width, item.y + item.height))
        if not any(min(box.x2, ui.x2) >= max(box.x1, ui.x1) and
                   min(box.y2, ui.y2) >= max(box.y1, ui.y1) for ui in excluded_boxes):
            result.append(item)
    return result


def _separate_river_outline(mask, oversized, size, config):
    """Recover one normal face joined to a thin highlight, keeping solid occlusions."""
    if not config.river_outline_opening_fraction:
        return [], oversized
    width, height = size
    minimum_side = min(width * config.river.width_fraction[0],
                       height * config.river.height_fraction[0])
    kernel_size = max(3, round(minimum_side * config.river_outline_opening_fraction) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    recovered, remaining = [], []
    for item in oversized:
        # Opening affects only the segmentation mask; CNN sees original pixels.
        local = mask[item.y:item.y+item.height, item.x:item.x+item.width]
        opened = cv2.morphologyEx(local, cv2.MORPH_OPEN, kernel,
                                 borderType=cv2.BORDER_CONSTANT, borderValue=0)
        _, _, stats, _ = cv2.connectedComponentsWithStats(opened)
        pieces = [_Component(item.x+int(s[0]), item.y+int(s[1]),
                             int(s[2]), int(s[3]), int(s[4])) for s in stats[1:]]
        faces, large = _filter_components(pieces, size, config.river,
                                          config.minimum_fill_ratio)
        if len(faces) == 1 and not large:
            recovered.extend(faces)
        else:
            remaining.append(item)
    return recovered, remaining


def _tile(crop: ROICrop, component: _Component, *, region: str,
          row=0, column=0, orientation="upright", face_down=False):
    box = crop.map_box_to_screen(Box(component.x, component.y,
                                     component.x + component.width,
                                     component.y + component.height))
    image = crop.image[component.y:component.y + component.height,
                       component.x:component.x + component.width].copy()
    return SegmentedTile(box, image, region, row, column, orientation, face_down)


def _split_merged_components(mask: np.ndarray, accepted: list[_Component],
                             oversized: list[_Component]):
    if len(accepted) < 3:
        return accepted, oversized
    typical_width = median(item.width for item in accepted)
    typical_height = median(item.height for item in accepted)
    remaining = []
    for item in oversized:
        width_ratio = item.width / typical_width
        height_ratio = item.height / typical_height
        parts = int(round(width_ratio))
        if (parts not in (2, 3) or abs(width_ratio - parts) > 0.4 or
                not 0.72 <= height_ratio <= 1.28):
            remaining.append(item)
            continue
        component_mask = mask[item.y:item.y + item.height, item.x:item.x + item.width]
        projection = np.count_nonzero(component_mask, axis=0)
        boundaries = [0]
        for part in range(1, parts):
            target = item.width * part / parts
            radius = max(2, round(typical_width * 0.22))
            start = max(boundaries[-1] + 1, round(target) - radius)
            stop = min(item.width - 1, round(target) + radius + 1)
            cut = (start + int(np.argmin(projection[start:stop]))
                   if stop > start else round(target))
            boundaries.append(cut)
        boundaries.append(item.width)
        split = []
        for start, stop in zip(boundaries, boundaries[1:]):
            section = component_mask[:, start:stop]
            points = cv2.findNonZero(section)
            if points is None:
                break
            x, y, width, height = cv2.boundingRect(points)
            area = int(np.count_nonzero(section[y:y + height, x:x + width]))
            split.append(_Component(item.x + start + x, item.y + y,
                                    width, height, area))
        if len(split) == parts:
            accepted.extend(split)
        else:
            remaining.append(item)
    return accepted, remaining


class HandSegmenter:
    def __init__(self, config: VisionConfig):
        self.config = config

    def segment(self, crop: ROICrop) -> SegmentationResult:
        _, components = _face_components(crop.image, self.config)
        height, width = crop.image.shape[:2]
        accepted, oversized = _filter_components(
            components, (width, height), self.config.hand, self.config.minimum_fill_ratio)
        accepted.sort(key=lambda item: item.center[0])
        issues = []
        if oversized:
            issues.append("Oversized white component may be animation or occlusion")
        if len(accepted) > self.config.hand.max_tiles:
            issues.append("Too many hand tile candidates")
        drawn_index = None
        if len(accepted) >= 3:
            gaps = [accepted[i + 1].center[0] - accepted[i].center[0]
                    for i in range(len(accepted) - 1)]
            baseline = median(gaps[:-1]) if len(gaps) > 1 else gaps[0]
            if baseline > 0 and gaps[-1] >= baseline * self.config.drawn_gap_ratio:
                drawn_index = len(accepted) - 1
        tiles = tuple(_tile(crop, item, region=crop.name, column=index)
                      for index, item in enumerate(accepted))
        return SegmentationResult(tiles, drawn_index, tuple(issues))


class RiverSegmenter:
    def __init__(self, config: VisionConfig):
        self.config = config

    def segment(self, crop: ROICrop, *, excluded_boxes=()) -> SegmentationResult:
        mask, components = _face_components(crop.image, self.config)
        components = _outside_ui_components(components, crop, excluded_boxes)
        height, width = crop.image.shape[:2]
        accepted, oversized = _filter_components(
            components, (width, height), self.config.river, self.config.minimum_fill_ratio)
        accepted, oversized = _split_merged_components(mask, accepted, oversized)
        recovered, oversized = _separate_river_outline(
            mask, oversized, (width, height), self.config)
        accepted.extend(_outside_ui_components(recovered, crop, excluded_boxes))
        issues = []
        if oversized:
            issues.append("Oversized white component may be animation or occlusion")
        if len(accepted) > self.config.river.max_tiles:
            issues.append("Too many river tile candidates")
        typical_height = median(item.height for item in accepted) if accepted else 0
        rows: list[list[_Component]] = []
        off_grid = []
        for item in sorted(accepted, key=lambda value: value.center[1]):
            if not rows:
                rows.append([item])
                continue
            row_center = median(value.center[1] for value in rows[-1])
            row_delta = item.center[1] - row_center
            if row_delta <= typical_height * self.config.row_tolerance_ratio:
                rows[-1].append(item)
            elif row_delta <= typical_height * self.config.maximum_row_gap_ratio:
                rows.append([item])
            else:
                off_grid.append(item)
        if off_grid:
            issues.append("Off-grid river components ignored")
        tiles = []
        for row_index, row in enumerate(rows):
            for column, item in enumerate(sorted(row, key=lambda value: value.center[0])):
                orientation = ("horizontal" if item.width / item.height >=
                               self.config.horizontal_aspect_ratio else "upright")
                tiles.append(_tile(crop, item, region=crop.name, row=row_index,
                                   column=column, orientation=orientation))
        return SegmentationResult(tuple(tiles), None, tuple(issues))


class TileRegionSegmenter:
    """Segment a small ROI containing zero or more visible tile faces."""

    def __init__(self, config: VisionConfig, limits: ComponentLimits):
        self.config = config
        self.limits = limits

    def segment(self, crop: ROICrop, *, excluded_boxes=()) -> SegmentationResult:
        _, components = _face_components(crop.image, self.config)
        components = _outside_ui_components(components, crop, excluded_boxes)
        height, width = crop.image.shape[:2]
        accepted, oversized = _filter_components(
            components, (width, height), self.limits, self.config.minimum_fill_ratio)
        accepted = [item for item in accepted
                    if item.height / item.width >=
                    self.config.upright_minimum_aspect_ratio]
        accepted.sort(key=lambda item: (item.center[0], item.center[1]))
        issues = []
        if oversized:
            issues.append("Oversized white component may be animation or occlusion")
        if len(accepted) > self.limits.max_tiles:
            issues.append("Too many tile candidates in region")
        tiles = tuple(_tile(crop, item, region=crop.name, column=index)
                      for index, item in enumerate(accepted))
        return SegmentationResult(tiles, None, tuple(issues))


class MeldSegmenter:
    """Segment visible meld tiles and retain their spatial group membership."""

    def __init__(self, config: VisionConfig, limits: ComponentLimits):
        self.config = config
        self.limits = limits

    def segment(self, crop: ROICrop) -> SegmentationResult:
        mask, components = _face_components(crop.image, self.config)
        height, width = crop.image.shape[:2]
        accepted, oversized = _filter_components(
            components, (width, height), self.limits, self.config.minimum_fill_ratio)
        accepted, oversized = _split_merged_components(mask, accepted, oversized)
        backs = []
        if self.config.back_hsv is not None:
            settings = self.config.back_hsv
            hsv = cv2.cvtColor(crop.image, cv2.COLOR_BGR2HSV)
            back_mask = cv2.inRange(hsv, settings.lower, settings.upper)
            _, _, stats, _ = cv2.connectedComponentsWithStats(back_mask)
            components = [_Component(*(int(v) for v in stat)) for stat in stats[1:]]
            backs, _ = _filter_components(components, (width, height), self.limits,
                                          self.config.back_minimum_fill_ratio)
            # Concealed opponent hands and walls also contain orange backs.
            # Retain only backs bracketing two face candidates in a compact run;
            # MeldReader still requires matching face identities before ankan.
            ordered = sorted(accepted + backs, key=lambda item: item.center[0])
            supported = []
            for start in range(max(0, len(ordered) - 3)):
                quartet = ordered[start:start+4]
                if [item in backs for item in quartet] != [True, False, False, True]:
                    continue
                spacing = median(item.width for item in quartet) * self.config.meld_group_gap_ratio
                if (all(0 < b.center[0] - a.center[0] <= spacing
                        for a, b in zip(quartet, quartet[1:]))):
                    supported.extend((quartet[0], quartet[-1]))
            backs = list(dict.fromkeys(supported))
        accepted.extend(backs)
        accepted.sort(key=lambda item: (item.center[0], item.center[1]))
        issues = []
        if oversized:
            issues.append("Oversized meld component may be animation or occlusion")
        if len(accepted) > self.limits.max_tiles:
            issues.append("Too many meld tile candidates")
        groups: list[list[_Component]] = []
        typical_width = median(item.width for item in accepted) if accepted else 0
        for item in accepted:
            if (not groups or item.center[0] - groups[-1][-1].center[0] >
                    typical_width * self.config.meld_group_gap_ratio):
                groups.append([item])
            else:
                groups[-1].append(item)
        if any(len(group) not in (3, 4) for group in groups):
            issues.append("Ambiguous meld grouping")
        tiles = []
        for group_index, group in enumerate(groups):
            for column, item in enumerate(group):
                orientation = ("horizontal" if item.width / item.height >=
                               self.config.horizontal_aspect_ratio else "upright")
                tiles.append(_tile(crop, item, region=crop.name, row=group_index,
                                   column=column, orientation=orientation,
                                   face_down=item in backs))
        return SegmentationResult(tuple(tiles), None, tuple(issues))
