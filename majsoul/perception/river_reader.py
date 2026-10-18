from __future__ import annotations

from .config import VisionConfig
from .roi_extractor import ROICrop, rotate_roi, outside_occlusions
from .schemas import RecognizedTile, TileSequenceReading
from .tile_classifier import TileTemplateClassifier
from .tile_segmenter import RiverSegmenter


class RiverReader:
    """Read one normalized screen-space river and retain declaration orientation."""

    def __init__(self, config: VisionConfig, classifier: TileTemplateClassifier):
        self.segmenter = RiverSegmenter(config)
        self.classifier = classifier

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
            issues.append(f"Unknown river tiles at indexes: {unknown}")
        return TileSequenceReading(crop.name, tiles, None, tuple(issues))
