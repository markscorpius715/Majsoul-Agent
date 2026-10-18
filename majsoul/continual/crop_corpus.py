from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re
from uuid import uuid4

import cv2
import numpy as np

from ..calibration.frames import read_image, write_image
from ..calibration.roi_config import ROIConfig
from ..perception import (HandSegmenter, MeldSegmenter, RiverSegmenter,
                          TileRegionSegmenter, VisionConfig, extract_roi,
                          rotate_roi)
from ..perception.schemas import SegmentedTile
from .collector import iter_video
from .config import ContinualConfig
from .detector import difference_hash, hamming_distance


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _slug(value: str) -> str:
    result = _SAFE_COMPONENT.sub("_", value).strip("._")
    if not result:
        raise ValueError("Crop region cannot be converted to a safe path")
    return result


def _thumbnail(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (48, 64), interpolation=cv2.INTER_AREA)


def _pixel_sha256(image: np.ndarray) -> str:
    header = f"{image.dtype}:{image.shape}".encode("ascii")
    return sha256(header + image.tobytes()).hexdigest()


@dataclass(frozen=True)
class ExtractedTileCrop:
    region: str
    position: str
    orientation: str
    tile: SegmentedTile
    issues: tuple[str, ...]


@dataclass(frozen=True)
class FrameTileCrops:
    crops: tuple[ExtractedTileCrop, ...]
    issues: tuple[str, ...]


class TileCropExtractor:
    """Extract visible tile faces without treating model predictions as labels."""

    def __init__(self, roi: ROIConfig, vision: VisionConfig, active_slots):
        self.roi = roi
        self.vision = vision
        self.active_slots = tuple(roi.layout_for_active_slots(active_slots))

    @staticmethod
    def _append(result: list[ExtractedTileCrop], region: str, segmentation) -> None:
        for tile in segmentation.tiles:
            position = f"r{tile.row}c{tile.column}"
            result.append(ExtractedTileCrop(
                region, position, tile.orientation, tile,
                tuple(segmentation.issues)))

    def extract(self, image: np.ndarray) -> FrameTileCrops:
        entries: list[ExtractedTileCrop] = []
        all_issues: list[str] = []

        hand = HandSegmenter(self.vision).segment(
            extract_roi(image, self.roi, "self_hand"))
        self._append(entries, "self_hand", hand)
        all_issues.extend(f"self_hand: {issue}" for issue in hand.issues)

        for slot in self.active_slots:
            river_region = f"river_{slot}"
            river = RiverSegmenter(self.vision).segment(rotate_roi(
                extract_roi(image, self.roi, river_region),
                self.vision.rotation_for_slot(slot)))
            self._append(entries, river_region, river)
            all_issues.extend(f"{river_region}: {issue}" for issue in river.issues)

            meld_region = f"melds_{slot}"
            meld = MeldSegmenter(
                self.vision, self.vision.meld_for_slot(slot)).segment(rotate_roi(
                    extract_roi(image, self.roi, meld_region),
                    self.vision.rotation_for_slot(slot)))
            self._append(entries, meld_region, meld)
            all_issues.extend(f"{meld_region}: {issue}" for issue in meld.issues)

        dora = TileRegionSegmenter(
            self.vision, self.vision.dora).segment(
                extract_roi(image, self.roi, "dora_indicators"))
        self._append(entries, "dora_indicators", dora)
        all_issues.extend(
            f"dora_indicators: {issue}" for issue in dora.issues)

        if self.roi.kita:
            for slot in self.active_slots:
                region = f"kita_{slot}"
                kita = TileRegionSegmenter(
                    self.vision, self.vision.kita_for_slot(slot)).segment(rotate_roi(
                        extract_roi(image, self.roi, region),
                        self.vision.rotation_for_slot(slot)))
                self._append(entries, region, kita)
                all_issues.extend(f"{region}: {issue}" for issue in kita.issues)

        return FrameTileCrops(tuple(entries), tuple(all_issues))


@dataclass
class _Representative:
    sample_index: int
    visual_hash: int
    thumbnail: np.ndarray


class CapturedTileCorpus:
    """Content-addressed, resumable store for unverified captured tile crops."""

    def __init__(
            self, output_root: str | Path, source_manifest: str | Path,
            *, roi: ROIConfig, active_slots, config: ContinualConfig,
            start: int, end: int, stride: int, resume: bool = False):
        if start < 0 or end <= start or stride < 1:
            raise ValueError("Invalid crop scan frame range or stride")
        self.root = Path(output_root)
        self.manifest_path = self.root / "manifest.json"
        self.source_manifest_path = Path(source_manifest)
        self.source = json.loads(
            self.source_manifest_path.read_text(encoding="utf-8-sig"))
        if self.source.get("source_type") != "captured_video":
            raise ValueError("Crop corpus requires a captured-video source manifest")
        video = self.source.get("video") or {}
        if video.get("player_count") != roi.player_count:
            raise ValueError("ROI game mode does not match source manifest")
        self.active_slots = tuple(roi.layout_for_active_slots(active_slots))
        self.config = config
        self.signature = {
            "source_id": self.source["source_id"],
            "player_count": video["player_count"],
            "mode": video.get("mode"),
            "active_slots": list(self.active_slots),
            "start": start,
            "end": end,
            "stride": stride,
            "dedup_hamming_distance": config.crop_dedup_hamming_distance,
            "dedup_mean_abs_difference":
                config.crop_dedup_mean_abs_difference,
            "minimum_frame_tiles": config.crop_minimum_frame_tiles,
        }
        self._index: dict[tuple[str, str], list[_Representative]] = {}
        self._hash_index: dict[tuple[str, str], dict[int, list[_Representative]]] = {}

        if self.manifest_path.exists():
            if not resume:
                raise ValueError(
                    "Crop corpus already exists; pass --resume to continue it")
            self.manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8-sig"))
            if self.manifest.get("scan_signature") != self.signature:
                raise ValueError("Existing crop corpus uses a different scan signature")
        else:
            self.manifest = {
                "schema_version": 1,
                "corpus_id": f"{self.source['source_id']}_tile_crops",
                "source_id": self.source["source_id"],
                "source_manifest": str(self.source_manifest_path),
                "domain": self.source.get("domain"),
                "player_count": video["player_count"],
                "mode": video.get("mode"),
                "label_status": "unverified",
                "training_allowed": False,
                "scan_signature": self.signature,
                "progress": {
                    "complete": False,
                    "last_frame": None,
                    "sampled_frames": 0,
                    "accepted_frames": 0,
                    "skipped_frames": 0,
                    "issue_frames": 0,
                    "candidate_crops": 0,
                    "unique_crops": 0,
                    "duplicate_crops": 0,
                    "updated_at": _utc_now(),
                },
                "samples": [],
                "notes": (
                    "All samples are unlabeled captured crops. Any classifier output "
                    "is review metadata, never human truth."),
            }
            self.root.mkdir(parents=True, exist_ok=True)
            self.checkpoint()
        self._load_index()

    @property
    def next_frame(self) -> int:
        last = self.manifest["progress"].get("last_frame")
        return self.signature["start"] if last is None else last + self.signature["stride"]

    @property
    def complete(self) -> bool:
        return bool(self.manifest["progress"].get("complete"))

    def _load_index(self) -> None:
        for index, sample in enumerate(self.manifest.get("samples", [])):
            path = self.root / sample["path"]
            image = read_image(path)
            representative = _Representative(
                index, int(sample["visual_hash"], 16), _thumbnail(image))
            key = (sample["region"], sample["orientation"])
            self._index.setdefault(key, []).append(representative)
            self._hash_index.setdefault(key, {}).setdefault(
                representative.visual_hash, []).append(representative)

    def _near_hashes(self, value: int):
        distance = self.config.crop_dedup_hamming_distance
        if distance > 2:
            return None
        values = [value]
        for changed in range(1, distance + 1):
            for bits in combinations(range(64), changed):
                candidate = value
                for bit in bits:
                    candidate ^= 1 << bit
                values.append(candidate)
        return values

    def _duplicate(self, key: tuple[str, str], visual_hash: int,
                   thumbnail: np.ndarray) -> _Representative | None:
        near_hashes = self._near_hashes(visual_hash)
        if near_hashes is None:
            candidates = self._index.get(key, ())
        else:
            buckets = self._hash_index.get(key, {})
            candidates = [item for value in near_hashes
                          for item in buckets.get(value, ())]
        for representative in candidates:
            if hamming_distance(representative.visual_hash, visual_hash) > \
                    self.config.crop_dedup_hamming_distance:
                continue
            difference = float(np.mean(cv2.absdiff(
                representative.thumbnail, thumbnail))) / 255.0
            if difference <= self.config.crop_dedup_mean_abs_difference:
                return representative
        return None

    def add_frame(self, batch: FrameTileCrops, *, frame_index: int,
                  timestamp: float) -> None:
        progress = self.manifest["progress"]
        progress["sampled_frames"] += 1
        progress["last_frame"] = frame_index
        if len(batch.crops) < self.config.crop_minimum_frame_tiles:
            progress["skipped_frames"] += 1
            self._maybe_checkpoint()
            return
        progress["accepted_frames"] += 1
        if batch.issues:
            progress["issue_frames"] += 1
        progress["candidate_crops"] += len(batch.crops)

        for extracted in batch.crops:
            image = extracted.tile.image
            visual_hash = difference_hash(image)
            thumbnail = _thumbnail(image)
            key = (extracted.region, extracted.orientation)
            duplicate = self._duplicate(key, visual_hash, thumbnail)
            if duplicate is not None:
                sample = self.manifest["samples"][duplicate.sample_index]
                sample["occurrence_count"] += 1
                sample["last_frame"] = frame_index
                sample["last_timestamp"] = timestamp
                progress["duplicate_crops"] += 1
                continue

            pixel_hash = _pixel_sha256(image)
            identity = sha256(
                f"{extracted.region}:{extracted.orientation}:{pixel_hash}"
                .encode("utf-8")).hexdigest()[:24]
            sample_id = f"crop_{identity}"
            relative = (Path("crops") / _slug(extracted.region) /
                        f"{sample_id}.png")
            target = self.root / relative
            if not target.exists():
                write_image(target, image)
            crop_hash = sha256(target.read_bytes()).hexdigest()
            tile = extracted.tile
            sample = {
                "id": sample_id,
                "source": self.source["source_id"],
                "player_count": self.source["video"]["player_count"],
                "mode": self.source["video"].get("mode"),
                "region": extracted.region,
                "position": extracted.position,
                "orientation": extracted.orientation,
                "first_frame": frame_index,
                "last_frame": frame_index,
                "first_timestamp": timestamp,
                "last_timestamp": timestamp,
                "occurrence_count": 1,
                "screen_box": [tile.box.x1, tile.box.y1,
                               tile.box.x2, tile.box.y2],
                "segmentation_issues": list(extracted.issues),
                "label": None,
                "verified_by": None,
                "split": "unassigned",
                "training_allowed": False,
                "pixel_sha256": pixel_hash,
                "crop_sha256": crop_hash,
                "visual_hash": f"{visual_hash:016x}",
                "path": relative.as_posix(),
            }
            sample_index = len(self.manifest["samples"])
            self.manifest["samples"].append(sample)
            representative = _Representative(
                sample_index, visual_hash, thumbnail)
            self._index.setdefault(key, []).append(representative)
            self._hash_index.setdefault(key, {}).setdefault(
                visual_hash, []).append(representative)
            progress["unique_crops"] += 1
        self._maybe_checkpoint()

    def _maybe_checkpoint(self) -> None:
        sampled = self.manifest["progress"]["sampled_frames"]
        if sampled % self.config.crop_checkpoint_frames == 0:
            self.checkpoint()

    def checkpoint(self, *, complete: bool | None = None) -> None:
        if complete is not None:
            self.manifest["progress"]["complete"] = complete
        self.manifest["progress"]["updated_at"] = _utc_now()
        _atomic_json(self.manifest_path, self.manifest)


def scan_captured_video(
        video: str | Path, *, source_manifest: str | Path,
        output_root: str | Path, roi: ROIConfig, vision: VisionConfig,
        active_slots, config: ContinualConfig, start: int = 0,
        end: int | None = None, stride: int = 15,
        resume: bool = False) -> dict:
    """Scan a complete timeline into an unverified, deduplicated crop corpus."""
    source = json.loads(Path(source_manifest).read_text(encoding="utf-8-sig"))
    total = int(source["video"]["frame_count"])
    stop = total if end is None else min(end, total)
    corpus = CapturedTileCorpus(
        output_root, source_manifest, roi=roi, active_slots=active_slots,
        config=config, start=start, end=stop, stride=stride, resume=resume)
    if corpus.complete:
        return corpus.manifest
    extractor = TileCropExtractor(roi, vision, active_slots)
    next_frame = corpus.next_frame
    if next_frame >= stop:
        corpus.checkpoint(complete=True)
        return corpus.manifest
    try:
        for image, frame_index, timestamp in iter_video(
                video, start=next_frame, end=stop, stride=stride):
            corpus.add_frame(
                extractor.extract(image), frame_index=frame_index,
                timestamp=timestamp)
    except BaseException:
        corpus.checkpoint(complete=False)
        raise
    corpus.checkpoint(complete=True)
    return corpus.manifest
