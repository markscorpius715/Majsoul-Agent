from __future__ import annotations

from .config import VisionConfig
from .roi_extractor import ROICrop
from .schemas import RecognizedTile, TileSequenceReading
from .tile_classifier import TileTemplateClassifier
from .tile_segmenter import HandSegmenter


class HandReader:
    """Segment and classify the visible concealed/drawn portion of self_hand."""

    def __init__(self, config: VisionConfig, classifier: TileTemplateClassifier):
        self.segmenter = HandSegmenter(config)
        self.classifier = classifier

    def read(self, crop: ROICrop) -> TileSequenceReading:
        segmentation = self.segmenter.segment(crop)
        predictions = self.classifier.classify_many(segmentation.tiles)
        tiles = tuple(RecognizedTile(detection, prediction)
                      for detection, prediction in zip(segmentation.tiles, predictions))
        issues = list(segmentation.issues)
        unknown = [index for index, prediction in enumerate(predictions)
                   if prediction.tile is None]
        if unknown:
            issues.append(f"Unknown hand tiles at indexes: {unknown}")
        return TileSequenceReading(crop.name, tiles, segmentation.drawn_index, tuple(issues))
