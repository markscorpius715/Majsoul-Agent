"""Human-reviewed continual perception infrastructure."""

from .config import ContinualConfig
from .collector import UnknownVideoCollector, build_known_index, iter_video
from .commands import parse_label_command, parse_review_command
from .dataset_audit import audit_parquet_splits, parse_split_specs
from .detector import (KnownVisualIndex, PersistentUnknownDetector,
                       assess_unknown, difference_hash, hamming_distance,
                       visual_embedding)
from .schemas import (Candidate, PerceptionLoopState, PersistenceResult, ReviewCase, ReviewKind,
                      ReviewStatus, UnknownAssessment, UnknownEvidence,
                      UnknownReason)
from .store import ReviewCaseStore, format_review_message
from .review_sheet import create_crop_corpus_sheet, create_review_sheet
from .video_source import register_captured_video
from .crop_corpus import (CapturedTileCorpus, FrameTileCrops,
                          TileCropExtractor, scan_captured_video)
from .corpus_review import CorpusReviewStore, CropReviewRecord
from .notifier import notify_review_case
from .label_gui import ReviewCaseIndex, ReviewRecord, preset_labels, tile_label_text
from .corpus_label_gui import discover_crop_manifests

__all__ = [
    "Candidate",
    "ContinualConfig",
    "KnownVisualIndex",
    "PersistenceResult",
    "PerceptionLoopState",
    "PersistentUnknownDetector",
    "ReviewCase",
    "ReviewCaseStore",
    "ReviewKind",
    "ReviewStatus",
    "UnknownAssessment",
    "UnknownEvidence",
    "UnknownReason",
    "UnknownVideoCollector",
    "assess_unknown",
    "build_known_index",
    "difference_hash",
    "format_review_message",
    "create_review_sheet",
    "create_crop_corpus_sheet",
    "CapturedTileCorpus",
    "CorpusReviewStore",
    "CropReviewRecord",
    "FrameTileCrops",
    "TileCropExtractor",
    "scan_captured_video",
    "notify_review_case",
    "ReviewCaseIndex",
    "ReviewRecord",
    "preset_labels",
    "tile_label_text",
    "discover_crop_manifests",
    "register_captured_video",
    "hamming_distance",
    "iter_video",
    "parse_label_command",
    "parse_review_command",
    "audit_parquet_splits",
    "parse_split_specs",
    "visual_embedding",
]
