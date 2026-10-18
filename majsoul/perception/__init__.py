"""Fixed-ROI perception components for the primary vision pipeline."""

from .config import VisionConfig
from .hand_reader import HandReader
from .glyph_classifier import GlyphTemplateClassifier, save_glyph_template
from .meld_reader import MeldReader
from .metadata_reader import GameMetadataReader, GlyphSegmenter, hsv_mask
from .layout_resolver import (LayoutCandidateEvaluation, SeatLayoutResolution,
                              SeatLayoutResolver)
from .river_reader import RiverReader
from .region_reader import TileRegionReader
from .roi_extractor import ROICrop, extract_roi, rotate_roi
from .schemas import (GameMetadataReading, GlyphPrediction, MeldReading,
                      NumericReading, RecognizedTile, RoundReading,
                      SeatReading, SegmentationResult, SegmentedGlyph,
                      SegmentedTile, TilePrediction, TileSequenceReading,
                      VisibleTilesReading)
from .tile_segmenter import (HandSegmenter, MeldSegmenter, RiverSegmenter,
                             TileRegionSegmenter)
from .tile_classifier import TileTemplateClassifier, save_template
from .tile_augment import PRESETS as TILE_AUGMENT_PRESETS
from .tile_augment import TileAugment, TileAugmentConfig
from .visible_tiles import VisibleTilesReader

__all__ = [
    "HandSegmenter",
    "HandReader",
    "GameMetadataReader",
    "GameMetadataReading",
    "GlyphPrediction",
    "GlyphSegmenter",
    "GlyphTemplateClassifier",
    "MeldReader",
    "MeldReading",
    "MeldSegmenter",
    "LayoutCandidateEvaluation",
    "NumericReading",
    "RecognizedTile",
    "RiverReader",
    "TileRegionReader",
    "TileRegionSegmenter",
    "RiverSegmenter",
    "RoundReading",
    "ROICrop",
    "SegmentationResult",
    "SeatReading",
    "SeatLayoutResolution",
    "SeatLayoutResolver",
    "SegmentedGlyph",
    "SegmentedTile",
    "TilePrediction",
    "TileSequenceReading",
    "TileAugment",
    "TileAugmentConfig",
    "TILE_AUGMENT_PRESETS",
    "TileTemplateClassifier",
    "VisionConfig",
    "VisibleTilesReader",
    "VisibleTilesReading",
    "extract_roi",
    "hsv_mask",
    "rotate_roi",
    "save_glyph_template",
    "save_template",
]
