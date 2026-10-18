from __future__ import annotations

from .config import VisionConfig
from .roi_extractor import ROICrop, rotate_roi
from .schemas import MeldReading, RecognizedTile, TilePrediction
from .tile_classifier import TileTemplateClassifier
from .tile_segmenter import MeldSegmenter
from ..tiles import tile_index


def partition_meld_tiles(tiles):
    """Return the unique legal partition of a spatial run; ambiguity stays unknown."""
    tiles = tuple(tiles)
    labels = [tile.prediction.tile for tile in tiles]
    indices = [tile_index(label) if label is not None else None for label in labels]

    def legal(values, detections):
        backs = [getattr(getattr(tile, 'detection', None), 'face_down', False)
                 for tile in detections]
        if any(backs):
            return (backs == [True, False, False, True] and
                    values[1] is not None and values[1] == values[2])
        if any(value is None for value in values):
            return False
        if len(set(values)) == 1:
            return True
        values = sorted(values)
        return (len(values) == 3 and values[-1] < 27
                and values[0] // 9 == values[-1] // 9
                and values == list(range(values[0], values[0]+3)))

    def solve(start):
        if start == len(tiles):
            return [()]
        results = []
        for size in (3, 4):
            end = start + size
            if end <= len(tiles) and legal(indices[start:end], tiles[start:end]):
                for rest in solve(end):
                    results.append((tiles[start:end], *rest))
                    if len(results) > 1:
                        return results
        return results

    solutions = solve(0)
    return solutions[0] if len(solutions) == 1 else None


class MeldReader:
    def __init__(self, config: VisionConfig, classifier: TileTemplateClassifier, slot: str):
        self.segmenter = MeldSegmenter(config, config.meld_for_slot(slot))
        self.classifier = classifier
        self.rotation_degrees = config.rotation_for_slot(slot)

    def read(self, crop: ROICrop) -> MeldReading:
        segmentation = self.segmenter.segment(rotate_roi(crop, self.rotation_degrees))
        faces = tuple(tile for tile in segmentation.tiles if not tile.face_down)
        face_predictions = iter(self.classifier.classify_many(faces))
        predictions = tuple(TilePrediction(None, 0.0, None, None, tile.orientation)
                            if tile.face_down else next(face_predictions)
                            for tile in segmentation.tiles)
        recognized = tuple(RecognizedTile(detection, prediction)
                           for detection, prediction in zip(segmentation.tiles, predictions))
        grouped = []
        for group_index in sorted({tile.detection.row for tile in recognized}):
            grouped.append(tuple(tile for tile in recognized
                                 if tile.detection.row == group_index))
        issues = list(segmentation.issues)
        unknown = [(tile.detection.row, tile.detection.column)
                   for tile in recognized if tile.prediction.tile is None and not tile.detection.face_down]
        if unknown:
            issues.append(f"Unknown meld tiles at positions: {unknown}")
        partitions = [partition_meld_tiles(group) for group in grouped]
        if all(partition is not None for partition in partitions):
            grouped = [group for partition in partitions for group in partition]
            issues = [issue for issue in issues if issue != 'Ambiguous meld grouping']
        kinds, opened = [], []
        for group in grouped:
            indices = [tile_index(tile.prediction.tile) for tile in group
                       if tile.prediction.tile is not None]
            if (len(group) == 4 and any(tile.detection.face_down for tile in group)
                    and partition_meld_tiles(group) is not None):
                kinds.append('ankan')
                opened.append(False)
            elif len(group) == 3 and partition_meld_tiles(group) is not None:
                kinds.append('pon' if len(set(indices)) == 1 else 'chi')
                opened.append(True)
            else:
                kinds.append('unknown')
                opened.append(True if len(group) == 3 or
                              any(tile.detection.orientation == 'horizontal'
                                  for tile in group) else None)
        if any(tile.detection.face_down for group, kind in zip(grouped, kinds)
               if kind != 'ankan' for tile in group):
            issues.append('Unresolved face-down tiles in meld region')
        return MeldReading(crop.name, tuple(grouped), tuple(issues),
                           tuple(kinds), tuple(opened))
