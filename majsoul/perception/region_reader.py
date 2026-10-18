from __future__ import annotations

from .config import ComponentLimits, VisionConfig
from .roi_extractor import ROICrop, rotate_roi, outside_occlusions
from .schemas import RecognizedTile, TileSequenceReading
from .tile_classifier import TileTemplateClassifier
from .tile_segmenter import TileRegionSegmenter


class TileRegionReader:
    """Read dora, kita, or another small sequence of visible tile faces."""

    def __init__(self, config: VisionConfig, classifier: TileTemplateClassifier,
                 limits: ComponentLimits, expected_tile: str | None = None):
        self.segmenter = TileRegionSegmenter(config, limits)
        self.classifier = classifier
        self.expected_tile = expected_tile

    def read(self, crop: ROICrop, rotation_degrees: int = 0, *, excluded_boxes=()) -> TileSequenceReading:
        segmentation = self.segmenter.segment(rotate_roi(crop, rotation_degrees),
                                               excluded_boxes=excluded_boxes)
        detections = outside_occlusions(segmentation.tiles, excluded_boxes)
        predictions = self.classifier.classify_many(detections)
        tiles = tuple(RecognizedTile(detection, prediction)
                      for detection, prediction in zip(detections, predictions))
        issues = list(segmentation.issues)
        if excluded_boxes:
            issues.append("UI occlusion; region unavailable")
        unknown = [index for index, prediction in enumerate(predictions)
                   if prediction.tile is None]
        if unknown:
            issues.append(f"Unknown tiles at indexes: {unknown}")
        if self.expected_tile is not None:
            unexpected = [index for index, prediction in enumerate(predictions)
                          if prediction.tile is not None and
                          prediction.tile != self.expected_tile]
            if unexpected:
                issues.append(
                    f"Expected {self.expected_tile} at indexes: {unexpected}")
        return TileSequenceReading(crop.name, tiles, None, tuple(issues))
